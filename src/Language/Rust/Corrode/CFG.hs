{-# LANGUAGE Rank2Types #-}
module Language.Rust.Corrode.CFG (
    Label, Terminator'(..), Terminator, BasicBlock(..),
    CFG(..), Unordered, DepthFirst, prettyCFG,
    BuildCFGT, mapBuildCFGT, addBlock, newLabel, buildCFG,
    removeEmptyBlocks, depthFirstOrder,
    prettyStructure, relooperRoot, structureCFG,
) where

import Control.Monad
import Control.Monad.Trans.State
import Data.Foldable
import qualified Data.IntMap.Lazy as IntMap
import qualified Data.IntSet as IntSet
import Data.Maybe
import Data.Traversable
import Text.PrettyPrint.HughesPJClass hiding (empty)
data BasicBlock s c = BasicBlock s (Terminator c)
type Label = Int
data Terminator' c l
    = Unreachable
    | Branch l
    | CondBranch c l l
    deriving Show
type Terminator c = Terminator' c Label

instance Functor (Terminator' c) where
    fmap = fmapDefault

instance Foldable (Terminator' c) where
    foldMap = foldMapDefault

instance Traversable (Terminator' c) where
    traverse _ Unreachable = pure Unreachable
    traverse f (Branch l) = Branch <$> f l
    traverse f (CondBranch c l1 l2) = CondBranch c <$> f l1 <*> f l2
data Unordered
data DepthFirst
data CFG k s c = CFG Label (IntMap.IntMap (BasicBlock s c))

instance (Show s, Show c) => Show (CFG k s c) where
    show = render . prettyCFG (text . show) (text . show)
prettyCFG :: (s -> Doc) -> (c -> Doc) -> CFG k s c -> Doc
prettyCFG fmtS fmtC (CFG entry blocks) = vcat $
    (text "start @" <> text (show entry)) : blocks'
    where
    blocks' = do
        (label, BasicBlock stmts term) <- IntMap.toList blocks
        let blockHead = text (show label) <> text ":"
        let blockBody = fmtS stmts
        let blockTail = case term of
                Unreachable -> text "// unreachable"
                Branch to -> text ("goto " ++ show to ++ ";")
                CondBranch cond true false ->
                    text "if(" <> fmtC cond
                        <> text ") goto " <> text (show true)
                        <> text "; else goto " <> text (show false)
                        <> text ";"
        blockHead : map (nest 4) [blockBody, blockTail] ++ [text ""]
type BuildCFGT m s c = StateT (BuildState s c) m
mapBuildCFGT
    :: (forall st. m (a, st) -> n (b, st))
    -> BuildCFGT m s c a -> BuildCFGT n s c b
mapBuildCFGT = mapStateT
data BuildState s c = BuildState
    { buildLabel :: Label
    , buildBlocks :: IntMap.IntMap (BasicBlock s c)
    }
newLabel :: Monad m => BuildCFGT m s c Label
newLabel = do
    old <- get
    put old { buildLabel = buildLabel old + 1 }
    return (buildLabel old)
addBlock :: Monad m => Label -> s -> Terminator c -> BuildCFGT m s c ()
addBlock label stmt terminator = do
    modify $ \ st -> st
        { buildBlocks = IntMap.insert label (BasicBlock stmt terminator)
            (buildBlocks st)
        }
buildCFG :: Monad m => BuildCFGT m s c Label -> m (CFG Unordered s c)
buildCFG root = do
    (label, final) <- runStateT root (BuildState 0 IntMap.empty)
    return (CFG label (buildBlocks final))
removeEmptyBlocks :: Foldable f => CFG k (f s) c -> CFG Unordered (f s) c
removeEmptyBlocks (CFG start blocks) = CFG (rewrite start) blocks'
    where
    go = do
        (empties, done) <- get
        case IntMap.minViewWithKey empties of
            Nothing -> return ()
            Just ((from, to), empties') -> do
                put (empties', done)
                step from to
                go
    step from to = do
        (empties, done) <- get
        case IntMap.splitLookup to empties of
            (_, Nothing, _) -> return ()
            (e1, Just to', e2) -> do
                put (e1 `IntMap.union` e2, done)
                step to to'
        (empties', done') <- get
        let to' = IntMap.findWithDefault to to done'
        put (empties', IntMap.insert from to' done')
    isBlockEmpty (BasicBlock s (Branch to)) | null s = Just to
    isBlockEmpty _ = Nothing
    rewrites = snd $ execState go (IntMap.mapMaybe isBlockEmpty blocks, IntMap.empty)
    rewrite to = IntMap.findWithDefault to to rewrites
    discards = IntMap.keysSet (IntMap.filterWithKey (/=) rewrites)
    rewriteBlock from _ | from `IntSet.member` discards = Nothing
    rewriteBlock _ (BasicBlock b term) = Just (BasicBlock b (fmap rewrite term))
    blocks' = IntMap.mapMaybeWithKey rewriteBlock blocks
data StructureLabel s c
    = GoTo { structureLabel :: Label }
    | ExitTo { structureLabel :: Label }
    | Nested [Structure s c]
    deriving Show

type StructureTerminator s c = Terminator' c (StructureLabel s c)
type StructureBlock s c = (s, StructureTerminator s c)

data Structure' s c a
    = Simple s (StructureTerminator s c)
    | Loop a
    | Multiple (IntMap.IntMap a) a
    deriving Show

data Structure s c = Structure
    { structureEntries :: IntSet.IntSet
    , structureBody :: Structure' s c [Structure s c]
    }
    deriving Show

prettyStructure :: (Show s, Show c) => [Structure s c] -> Doc
prettyStructure = vcat . map go
    where
    go (Structure _ (Simple s term)) = text (show s ++ ";") $+$ text (show term)
    go (Structure entries (Loop body)) = prettyGroup entries "loop" (prettyStructure body)
    go (Structure entries (Multiple handlers unhandled)) = prettyGroup entries "match" $
        vcat [ text (show entry ++ " =>") $+$ nest 2 (prettyStructure handler) | (entry, handler) <- IntMap.toList handlers ]
        $+$ if null unhandled then mempty else (text "_ =>" $+$ nest 2 (prettyStructure unhandled))

    prettyGroup entries kind body =
        text "{" <> hsep (punctuate (text ",") (map (text . show) (IntSet.toList entries))) <> text ("} " ++ kind)
        $+$ nest 2 body

relooperRoot :: Monoid s => CFG k s c -> [Structure s c]
relooperRoot (CFG entry blocks) = relooper (IntSet.singleton entry) $
    IntMap.map (\ (BasicBlock s term) -> (s, fmap GoTo term)) blocks

relooper :: Monoid s => IntSet.IntSet -> IntMap.IntMap (StructureBlock s c) -> [Structure s c]
relooper entries blocks =
    let (returns, noreturns) = partitionMembers entries $ IntSet.unions $ map successors $ IntMap.elems blocks
        (present, absent) = partitionMembers entries (IntMap.keysSet blocks)
    in case (IntSet.toList noreturns, IntSet.toList returns) of
    ([], []) -> []
    ([entry], []) -> case IntMap.updateLookupWithKey (\ _ _ -> Nothing) entry blocks of
        (Just (s, term), blocks') -> Structure
            { structureEntries = entries
            , structureBody = Simple s term
            } : relooper (successors (s, term)) blocks'
        (Nothing, _) -> Structure
            { structureEntries = entries
            , structureBody = Simple mempty (Branch (GoTo entry))
            } : []
    _ | not (IntSet.null absent) ->
        if IntSet.null present then [] else Structure
            { structureEntries = entries
            , structureBody = Multiple
                (IntMap.fromSet (const []) absent)
                (relooper present blocks)
            } : []
    ([], _) -> Structure
        { structureEntries = entries
        , structureBody = Loop (relooper entries blocks')
        } : relooper followEntries followBlocks
        where
        returns' = (strictReachableFrom `IntMap.intersection` blocks) `restrictKeys` entries
        bodyBlocks = blocks `restrictKeys`
            IntSet.unions (IntMap.keysSet returns' : IntMap.elems returns')
        followBlocks = blocks `IntMap.difference` bodyBlocks
        followEntries = outEdges bodyBlocks
        markEdge (GoTo label)
            | label `IntSet.member` (followEntries `IntSet.union` entries)
            = ExitTo label
        markEdge edge = edge
        blocks' = IntMap.map (\ (s, term) -> (s, fmap markEdge term)) bodyBlocks
    _ -> Structure
        { structureEntries = entries
        , structureBody = Multiple handlers unhandled
        } : relooper followEntries followBlocks
        where
        reachableFrom = IntMap.unionWith IntSet.union (IntMap.fromSet IntSet.singleton entries) strictReachableFrom
        singlyReached = flipEdges $ IntMap.filter (\ r -> IntSet.size r == 1) $ IntMap.map (IntSet.intersection entries) reachableFrom
        handledEntries = IntMap.map (\ within -> blocks `restrictKeys` within) singlyReached
        unhandledEntries = entries `IntSet.difference` IntMap.keysSet handledEntries
        handledBlocks = IntMap.unions (IntMap.elems handledEntries)
        followBlocks = blocks `IntMap.difference` handledBlocks
        followEntries = unhandledEntries `IntSet.union` outEdges handledBlocks
        makeHandler entry blocks' = relooper (IntSet.singleton entry) blocks'
        allHandlers = IntMap.mapWithKey makeHandler handledEntries
        (unhandled, handlers) = if IntMap.keysSet allHandlers == entries
            then
                let (lastHandler, otherHandlers) = IntMap.deleteFindMax allHandlers
                in (snd lastHandler, otherHandlers)
            else ([], allHandlers)

    where
    strictReachableFrom = flipEdges (go (IntMap.map successors blocks))
        where
        grow r = IntMap.map (\ seen -> IntSet.unions $ seen : IntMap.elems (r `restrictKeys` seen)) r
        go r = let r' = grow r in if r /= r' then go r' else r'

restrictKeys :: IntMap.IntMap a -> IntSet.IntSet -> IntMap.IntMap a
restrictKeys m s = m `IntMap.intersection` IntMap.fromSet (const ()) s

outEdges :: IntMap.IntMap (StructureBlock s c) -> IntSet.IntSet
outEdges blocks = IntSet.unions (map successors $ IntMap.elems blocks) `IntSet.difference` IntMap.keysSet blocks

partitionMembers :: IntSet.IntSet -> IntSet.IntSet -> (IntSet.IntSet, IntSet.IntSet)
partitionMembers a b = (a `IntSet.intersection` b, a `IntSet.difference` b)

successors :: StructureBlock s c -> IntSet.IntSet
successors (_, term) = IntSet.fromList [ target | GoTo target <- toList term ]

flipEdges :: IntMap.IntMap IntSet.IntSet -> IntMap.IntMap IntSet.IntSet
flipEdges edges = IntMap.unionsWith IntSet.union [ IntMap.fromSet (const (IntSet.singleton from)) to | (from, to) <- IntMap.toList edges ]
simplifyStructure :: Monoid s => [Structure s c] -> [Structure s c]
simplifyStructure = foldr go [] . map descend
    where
    descend structure = structure { structureBody =
        case structureBody structure of
        Simple s term -> Simple s term
        Multiple handlers unhandled ->
            Multiple (IntMap.map simplifyStructure handlers) (simplifyStructure unhandled)
        Loop body -> Loop (simplifyStructure body)
    }
    go (Structure entries (Simple s term))
       (Structure _ (Multiple handlers unhandled) : rest) =
        Structure entries (Simple s (fmap rewrite term)) : rest
        where
        rewrite (GoTo to) = Nested
            $ Structure (IntSet.singleton to) (Simple mempty (Branch (GoTo to)))
            : IntMap.findWithDefault unhandled to handlers
        rewrite _ = error ("simplifyStructure: Simple/Multiple invariants violated in " ++ show entries)

    go block rest = block : rest

-- We no longer care about ordering, but reachability needs to only include
-- nodes that are reachable from the function entry, and this has the side
-- effect of pruning unreachable nodes from the graph.
depthFirstOrder :: CFG k s c -> CFG DepthFirst s c
depthFirstOrder (CFG start blocks) = CFG start' blocks'
    where
    search label = do
        (seen, order) <- get
        unless (label `IntSet.member` seen) $ do
            put (IntSet.insert label seen, order)
            case IntMap.lookup label blocks of
                Just (BasicBlock _ term) -> traverse_ search term
                _ -> return ()
            modify (\ (seen', order') -> (seen', label : order'))
    final = snd (execState (search start) (IntSet.empty, []))
    start' = 0
    mapping = IntMap.fromList (zip final [start'..])
    rewrite label = IntMap.findWithDefault (error "basic block disappeared") label mapping
    rewriteBlock label (BasicBlock body term) = (label, BasicBlock body (fmap rewrite term))
    blocks' = IntMap.fromList (IntMap.elems (IntMap.intersectionWith rewriteBlock mapping blocks))
structureCFG
    :: Monoid s
    => (Maybe Label -> s)
    -> (Maybe Label -> s)
    -> (Label -> s -> s)
    -> (c -> s -> s -> s)
    -> (Label -> s)
    -> ([(Label, s)] -> s -> s)
    -> CFG DepthFirst s c
    -> (Bool, s)
structureCFG mkBreak mkContinue mkLoop mkIf mkGoto mkMatch cfg =
    (hasMultiple root, foo [] mempty root)
    where
    root = simplifyStructure (relooperRoot cfg)
    foo exits next' = snd . foldr go (next', mempty)
        where
        go structure (next, rest) = (structureEntries structure, go' structure next `mappend` rest)

        go' (Structure entries (Simple body term)) next = body `mappend` case term of
                Unreachable -> mempty
                Branch to -> branch to
                CondBranch c t f -> mkIf c (branch t) (branch f)
            where
            branch (Nested nested) = foo exits next nested
            branch to | structureLabel to `IntSet.member` next =
                insertGoto (structureLabel to) (next, mempty)
            branch (ExitTo to) | isJust target = insertGoto to (fromJust target)
                where
                inScope immediate (label, local) = do
                    (follow, mkStmt) <- IntMap.lookup to local
                    return (follow, mkStmt (immediate label))
                target = msum (zipWith inScope (const Nothing : repeat Just) exits)
            branch to = error ("structureCFG: label " ++ show (structureLabel to) ++ " is not a valid exit from " ++ show entries)

            insertGoto _ (target, s) | IntSet.size target == 1 = s
            insertGoto to (_, s) = mkGoto to `mappend` s

        go' (Structure _ (Multiple handlers unhandled)) next =
            mkMatch [ (label, foo exits next body) | (label, body) <- IntMap.toList handlers ] (foo exits next unhandled)

        go' (Structure entries (Loop body)) next = mkLoop label (foo exits' entries body)
            where
            label = IntSet.findMin entries
            exits' =
                ( label
                , IntMap.union
                    (IntMap.fromSet (const (entries, mkContinue)) entries)
                    (IntMap.fromSet (const (next, mkBreak)) next)
                ) : exits

hasMultiple :: [Structure s c] -> Bool
hasMultiple = any (go . structureBody)
    where
    go (Multiple{}) = True
    go (Simple _ term) = or [ hasMultiple nested | Nested nested <- toList term ]
    go (Loop body) = hasMultiple body
