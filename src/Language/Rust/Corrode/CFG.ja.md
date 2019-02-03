制御フローのためのさまざまなプリミティブを持つ言語は、自動翻訳のためには注意が必要です。 Cのような任意のgoto文を許す言語から、他の広く使われているプログラミング言語のように許されない言語へ翻訳しているのであれば、特にそうです。

このモジュールは2つのステップでその複雑さの大部分を引き受けます。

1.まず、ソースプログラム内の関数のすべてのループ、条件式、および問題点を表す制御フローグラフ（CFG）を作成できます。 （これは通常かなり簡単です。）

2.そしてこのモジュールはそのCFGを分析し、どの部分をループとして扱い、どの部分を `if`ステートメントとして扱うべきか、そしてそれらが変換された関数に対して現れるべき順序を識別することができます。

ソースに `goto`ステートメントがある場合、ステップ2の出力はステップ1への入力とは非常に異なって見えるかもしれません。

```haskell
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
```


制御フローグラフ表現
================================

制御フローグラフは、シーケンシャルコードを含む「基本ブロック」と、コンピュータが現在の基本ブロックの最後に到達したときに次に実行するものを示す矢印の集まりです。

有効な基本ブロックになるには、制御フローはブロックの始めにのみ入り、終わりにのみ出る必要があります。

基本ブロックには、基本ブロック内のコードを表すために使用する型に関係なく、型パラメータ `s`があります。
このモジュールは一般的にあなたがどの表現を使用するか気にしません;あなたが選択したものは何でも、合理的な選択はステートメントのリストであるかもしれませんが、おそらく `Foldable`と` Monoid`の両方のインスタンスを持つべきです。
そうでなければ、このモジュールが提供するいくつかの重要な機能を使用することはできません。

（次に、 `c`型パラメータについて説明しながらターミネータについて説明します。）

すべての基本ブロックに、制御フローグラフの他の場所から参照するために使用できる任意の「ラベル」を割り当てます。
これは何でも構いませんが、ラベルとして個別の整数を使用すると便利です。

```haskell
data BasicBlock s c = BasicBlock s (Terminator c)
type Label = Int
```

基本ブロックは、次に進むブロックの指定で終わります。これをブロックの「ターミネータ」と呼びます。

これらのケースをモデル化します。

 -  `Unreachable`はソース言語がコントロールがこのブロックの終わりに到達しないことを保証することを示します。
これは通常、ブロックが `return`文で終わっているからです。
しかし、例えば、ブロックが、決して戻らないことがわかっている関数への呼び出しで終了した場合にも発生する可能性があります。

 -  `Branch`は、このブロックが完了すると、制御が常に指定されたブロックに進むことを示します。

 -  `CondBranch`は「条件付きブランチ」です。
指定された条件が実行時に真の場合、制御は最初に指定されたブロックに進みます。それ以外の場合は2番目のブロックに進みます。
条件付き分岐は、常に「真」の場合と「偽」の場合の両方を持つものとして表すことに注意してください。たとえばアセンブリ言語での条件付きジャンプで見られるような、暗黙の「フォールスルー」動作はありません。

```haskell
data Terminator' c l
    = Unreachable
    | Branch l
    | CondBranch c l l
    deriving Show
```

上記の `Terminator '型には2つのジェネリック型パラメータがあります。

1つ目は、条件式に使用する型です。
これはおそらくあなたのターゲット言語でブール式を表現するのに使用するどんな型でもあるべきですが、このモジュールはそれらの条件式の中にあるものをまったく見ないので、あなたはあなたが望む表現を使用できます。

2番目のタイプパラメータは、基本ブロックのラベルに使用するタイプに関係ありません。
上記では特定の `Label`型を選択しましたが、出て行く辺への一般的なアクセスのために標準の` Functor`と `Foldable`型クラスのインスタンスを定義できるようにこれを型パラメータにすると便利です。

便宜上、ラベルタイプが特に上で選択された `Label`であることを指定するタイプエイリアスを定義します。

```haskell
type Terminator c = Terminator' c Label

instance Functor (Terminator' c) where
    fmap = fmapDefault

instance Foldable (Terminator' c) where
    foldMap = foldMapDefault

instance Traversable (Terminator' c) where
    traverse _ Unreachable = pure Unreachable
    traverse f (Branch l) = Branch <$> f l
    traverse f (CondBranch c l1 l2) = CondBranch c <$> f l1 <*> f l2
```

これで、前のタイプに関して完全な制御フローグラフを定義できます。
それは "開始"ラベルを持ち、どの基本ブロックが関数の入り口で実行を開始する最初のものであるかを示します。ラベルからそれに対応する基本ブロックへのマップ。

CFGが構築された後、基本ブロックを有用な順序にソートするために私たちがする前処理ステップがあります。
ここでは、ソートが行われたかどうかを示すために、ここでは小さな型システムのトリックを使用します。
`CFG k`を受け入れる関数はブロックがソートされているかどうかを気にしません。
そのため、以下に注意してください。型シグネチャは重要な前提条件の文書として役立つためです。

この型システムのトリックにより、Haskellコンパイラは、呼び出し側がソートされたCFGのみをそれらを必要とする関数に渡すことを強制します。これは良い健全性チェックです。
ただし、このモジュール内では、CFGが実際にソートされている場合にのみソートされているとタグ付けし、必要に応じてソートされたCFGが必要であるとして関数にタグ付けするよう注意してください。
Haskellは魔法でそれを理解することはできません！

```haskell
data Unordered
data DepthFirst
data CFG k s c = CFG Label (IntMap.IntMap (BasicBlock s c))

instance (Show s, Show c) => Show (CFG k s c) where
    show = render . prettyCFG (text . show) (text . show)
```

問題が発生した場合は、人間が読める形式の制御フローグラフ全体を印刷できるので便利です。
この関数は、ステートメントと条件式をそれぞれフォーマットするためのヘルパー関数を取り、それらを各基本ブロック内で使用して、制御フローグラフ全体をフォーマットします。

```haskell
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
```


CFGの構築
=================

このモジュールは、制御フローグラフを作成するための小さなモナドインタフェースを提供します。
これは「モナド変換子」として提供されています。つまり、このモナドを他のモナドと組み合わせることができます。
例えば、ステートメントや式を正しく翻訳するためにスコープ内にある変数宣言についての情報を保持する必要があるなら、そのために `State`モナドを使い、その上にこの` BuildCFGT`モナドを重ねることができます。
その後、必要に応じてどちらかのモナドのアクションを使用できます。

```haskell
type BuildCFGT m s c = StateT (BuildState s c) m
```

これはモナド変換子なので、基礎となるモナドを変換する操作を実行する必要があるかもしれません。
たとえば、 `Reader`モナドは、その環境内で外側の計算が使うものとは異なる値で、そして` Writer`モナドの `listen`や` censor`オペレーションと同様に、ある特定のモナディックアクションを実行する `local`オペレーションを持ちます。
これらの種類の操作を `BuildCFGT`で使うためには、それらを` mapBuildCFGT`でラップする必要があります。

ここでの型シグネチャは少し変です。
CFGビルダーの現在の状態を保持する必要があります。これは、通常、その状態を舞台裏で運ぶ通常のモナド規則を中断しているためです。
しかし、その過程でCFGビルダーの状態を覗いたり変更したりすることはできません。
これはGHCの `Rank2Types`言語拡張（このモジュールの一番上で有効になっている）を使って、あなたの変換がすべての可能な状態型に対して機能しなければならないことを宣言することによって実施されます。変わらずにデータ。

```haskell
mapBuildCFGT
    :: (forall st. m (a, st) -> n (b, st))
    -> BuildCFGT m s c a -> BuildCFGT n s c b
mapBuildCFGT = mapStateT
```

新しい制御フローグラフを作成する間、2つのことを追跡する必要があります。これまでに作成された基本ブロックと、次の基本ブロックに使用する一意のラベルです。
両方とも新しいデータ型 `BuildState`に入れます。

固有のラベル用に別のカウンターを用意する必要がないように思われるかもしれません。
構築された最後のブロックのラベルを見て、1を加えて、それを次のブロックのラベルとして使用することはできませんか。

残念ながら、CFGを構築する際には、まだ構築していないブロックを参照する必要があります。
たとえば、ループを構築するには、ループヘッダーに戻る分岐を使用してループの本体を構築し、次に本体への分岐を使用してループヘッダーを構築することがあります。

つまり、対応するブロックを完成させる前に、ラベルをいくつでも生成しなければならない可能性があるため、どのIDをすでに配布したかを追跡する必要があります。

これはまた、まだ構築されていない他のブロックに分岐するブロックを含むため、このCFGの中間表現は一般に有効なCFGではないことを意味します。
すべてのブロックが最終的に追加されることを保証するのは呼び出し側の責任です。

```haskell
data BuildState s c = BuildState
    { buildLabel :: Label
    , buildBlocks :: IntMap.IntMap (BasicBlock s c)
    }
```

`newLabel`は一意の` Label`を返すだけです。

```haskell
newLabel :: Monad m => BuildCFGT m s c Label
newLabel = do
    old <- get
    put old { buildLabel = buildLabel old + 1 }
    return (buildLabel old)
```

`addBlock`は与えられたステートメントとターミネータを状態に保存します。

```haskell
addBlock :: Monad m => Label -> s -> Terminator c -> BuildCFGT m s c ()
addBlock label stmt terminator = do
    modify $ \ st -> st
        { buildBlocks = IntMap.insert label (BasicBlock stmt terminator)
            (buildBlocks st)
        }
```

最後に、ビルダーを実行し、それが構築したCFGを返す関数があります。
Builderの戻り値は、制御フロー・グラフの入り口点として使用するラベルでなければなりません。

まだソートされていないので、構築されたCFGは `Unordered`としてタグ付けされていることに注意してください。

```haskell
buildCFG :: Monad m => BuildCFGT m s c Label -> m (CFG Unordered s c)
buildCFG root = do
    (label, final) <- runStateT root (BuildState 0 IntMap.empty)
    return (CFG label (buildBlocks final))
```

CFGを構築するための簡単な翻訳を書くのは普通のことです。
たとえば、文を含まず、無条件に別の場所に分岐するだけの基本ブロックを多数作成することがあります。
CFGの意味を変えることなく、少し注意深くすれば、これらのブロックを安全に削除することができます。それが `removeEmptyBlocks`です。

> ** NOTE **：これは必要だとは思わない。以下のアルゴリズムはすべて、私が理解できる限り、空のブロックが存在しても同じ出力を生成するはずです。
>しかし、何か問題が発生してエラーを報告する必要がある場合は、より簡単なCFGを調べるのがいいでしょう。
>だから私はこれを削除していませんが、それが重要ではないのでそれがどのように動作するかを文書化することを気にするつもりはありません。

> ** TODO **：関数を複雑にしすぎずにこれを `depthFirstOrder`に折り畳むことができるかどうか考えてください。

```haskell
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
```


CFGから構造化プログラムへの変換
===========================================

CFGを構築した後の本当の課題は、基本的なブロックの乱雑な積み重ねを構造化された制御フローに戻すことです。

この実装は、かなり多種多様な言語で機能します。
ターゲット言語は次のように仮定します。

1. If-then-else、
ループ、
3.マルチレベルループから抜けます。

その最後の点は説明が必要です。
ループを含むほとんどの言語では、プログラマがループから早く抜け出したり、現在の反復を終了せずにループの最初から再開したりする方法が提供されています。
（両方の種類の制御フローを「ループ出口」と呼びましょう。）これらの言語の多くで、ループ名を指定し、名前でループを指定することによって、プログラマーは一度に複数のループを終了できます。 。
このコードはあなたのターゲット言語が後者の種類の一つであると仮定しています。

```haskell
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
```

最初に、エントリラベルを、あるブロックが分岐する可能性があるものに対して、あるブロックが分岐する可能性があるものに分割します。
重要な考え方は、エントリラベルを出力の早い段階に配置する必要があるということですが、後で何かを分岐できる場合は、それらをループでラップして、制御フローをエントリポイントに戻すことができます。

これらの場合はそれぞれ、少なくとも1回の再帰呼び出しを行います。
このアルゴリズムが無限ループに陥らないようにするために、すべての再帰呼び出しが解決するための「より単純な」問題を持つようにする必要があります。 。
それぞれの場合において、副問題が本当に単純であることを示します。

```haskell
    let (returns, noreturns) = partitionMembers entries $ IntSet.unions $ map successors $ IntMap.elems blocks
        (present, absent) = partitionMembers entries (IntMap.keysSet blocks)
    in case (IntSet.toList noreturns, IntSet.toList returns) of
```

エントリポイントがない場合、前のブロックは残りのブロックに到達できないため、それらのコードを生成する必要はありません。
これが、このアルゴリズムの主な再帰ベースケースです。

```haskell
    ([], []) -> []
```

単純ブロック
-------------

ラベルが1つしかなく、それが現在のブロックセット内のブランチのターゲットではない場合は、そのラベルを出力の次に配置します。

この場合、再帰呼び出しを行う前に常に1ブロックが考慮から除外されるため、サブ問題は1ブロック小さくなります。

```haskell
    ([entry], []) -> case IntMap.updateLookupWithKey (\ _ _ -> Nothing) entry blocks of
        (Just (s, term), blocks') -> Structure
            { structureEntries = entries
            , structureBody = Simple s term
            } : relooper (successors (s, term)) blocks'
```

ターゲットが、後でどこかに配置することをすでに決めているブロックである場合は、コードジェネレータにcurrent-block state変数を適切に設定するように指示する偽のブロックを作成する必要があります。

```haskell
        (Nothing, _) -> Structure
            { structureEntries = entries
            , structureBody = Simple mempty (Branch (GoTo entry))
            } : []
```

後で配置されたブロックにスキップする
-------------------------------

複数のエントリラベルがあり、それらの一部または全部が後でどこかに配置することを決定したブロックを参照する場合、制御フローが実際にこれらのブロックを配置する場所に到達するまで、介在コードをスキップする方法が必要です。
（以前に配置したブロックに分岐する必要がある場合は、そのためのループが既に構築されているので、ここでそのケースを処理する必要はありません。）

これを実現するために、存在しないエントリラベルごとに空のブランチと、スキップしたいコードを含むelseブランチを含むMultipleブロックを構築します。
これにより、囲んでいるブロックの終わりまで制御フローが取得されます。
ターゲットブロックも存在しない場合は、ターゲットラベルを含むブロックに到達するまで、その時点で再度これを実行します。

しかし、else分岐に配置するコードがなければ、この手続きは何もしない `Multiple`ブロックを生成するので、その場合は何も発行しないようにすることができます。

```haskell
    _ | not (IntSet.null absent) ->
        if IntSet.null present then [] else Structure
            { structureEntries = entries
            , structureBody = Multiple
                (IntMap.fromSet (const []) absent)
                (relooper present blocks)
            } : []
```

ループ
-----

すべての入口ラベルがどこかのブロック内の分岐のターゲットである場合は、それらすべてのラベルを入口点としてループを作成します。

生成されたコードを単純にするために、ループを構築する前に、存在しないエントリ（以前のケース）を排除したいと思います。
エントリポイントが存在しないループを生成する場合、ループ内のエントリポイントを処理するには、ループから抜け出す必要があります。
代わりにこの順序でそれを行うことによって、不在のブランチのためにハンドラにコードをまったく必要としません。

この場合、ループ本体に対する再帰呼び出しと、ループの後に続くラベルに対する再呼び出しがあります。

 - ループ本体は同じエントリラベルを持ちます。
しかし、再帰呼び出しでは、ループになったすべてのブランチを削除するので、同じエントリラベルのセットでこのケースが再び発生しないことが保証されています。
他のケースでブロック数が減っている限り、これで終わりです。

 - ループに続くラベルについては、少なくとも現在のエントリラベルを考慮から除外したので、まだ構造化する必要があるブロックが少なくなります。

```haskell
    ([], _) -> Structure
        { structureEntries = entries
        , structureBody = Loop (relooper entries blocks')
        } : relooper followEntries followBlocks
        where
```

このループの本体に含めるべきラベルは、最終的にループのエントリポイントの1つに戻ってくる可能性があるすべてのラベルです。

`IntMap.keysSetは '== entries`を返します。
あるエントリが他のどのエントリからもアクセスできない場合は、最初に「Multiple」ブロックに分割します。

```haskell
        returns' = (strictReachableFrom `IntMap.intersection` blocks) `restrictKeys` entries
        bodyBlocks = blocks `restrictKeys`
            IntSet.unions (IntMap.keysSet returns' : IntMap.elems returns')
```

どのラベルがループ本体に属しているのかを特定したので、現在のブロックをループの内側にあるものとそれに続くものに分割することができます。

```haskell
        followBlocks = blocks `IntMap.difference` bodyBlocks
```

このループの内側から外側へ向かう分岐は、このループの後に続くブロックのエントリポイントを形成します。
（プログラムの早い段階で分岐することはできません。なぜなら、この分岐を囲むループに再帰する前に既に分岐を削除しているからです。）

```haskell
        followEntries = outEdges bodyBlocks
```

この時点で私たちはいくつかのブランチを `break`（それは` followEntries`の中にあります）または `continue`（それはこのループのエントリポイントの1つだったので）ブランチとして識別しました。
このループの本体を再構築するときに、それらの分岐を再び考慮してはいけません。そのため、ループ内のすべてのブロックの後続からそれらを削除します。

このループブロックのための `structureEntries`は` continue`エッジであるラベルを記録し、それに続くブロックのための `structureEntries`は` break`エッジであるラベルを記録するので、ここで追加情報を記録する必要はありません。

ループエントリへの分岐を削除できなかった場合は、再帰したときに別の `Loop`ブロックが生成されます。これはアルゴリズムが終了しないことを意味します。

ループから出るブランチを削除できなかったとしても、結果はまだ正しいと思いますが、必要以上に多くの `Multiple`ブロックがあるでしょう。

```haskell
        markEdge (GoTo label)
            | label `IntSet.member` (followEntries `IntSet.union` entries)
            = ExitTo label
        markEdge edge = edge
        blocks' = IntMap.map (\ (s, term) -> (s, fmap markEdge term)) bodyBlocks
```

複数項目ブロック
---------------------

それ以外の場合は、この時点で複数の制御フローパスをマージする必要があります。どのパスがオンになっているかを動的に確認するコードを作成することによってです。

`Multiple`ブロックでは、安全に分割できるように各エントリラベルに対して別々のハンドラを構築します。
各ハンドラに対して再帰的な呼び出しを行い、このブロックで処理できなかったすべてのブロックに対してもう1回呼び出します。

 - 未処理のブロックがある場合、各ハンドラには最初に起動したブロックよりも少ないブロックが含まれます。
すべてのエントリラベルを処理できた場合は、ブロックを少なくとも2つの空でないグループに分割したので、各グループは最初のサイズよりも必ず小さくなります。
ノーリターンエントリがない場合はループを作成したことになり、ノーリターンエントリが1つだけでリターン可能なエントリがない場合はエントリラベルが2つあることが必要です。単純なブロックを構築しました。

 - 各ハンドラは少なくともそのエントリラベルを消費するので、少なくとも1つのハンドラを生成する限り、未処理ブロックの再帰呼び出しのサブ問題は小さくなります。
他のどのエントリーラベルも、一連の分岐を通じてこのラベルに分岐できない場合にのみ、エントリーラベルを処理できます。
しかし、上記のようにループを構築することはできないので、少なくとも1つのエントリラベルには分岐がないことがわかります。したがって、このパスでは少なくとも1ブロックを消費することが保証されています。

```haskell
    _ -> Structure
        { structureEntries = entries
        , structureBody = Multiple handlers unhandled
        } : relooper followEntries followBlocks
        where
```

`singlyReached`マップの要素は互いに素な集合です。
証明： `IntMap`のキーは定義によって区別され、` filter`の後の値はシングルトン集合です。そのため、 `flipEdges`の後では、それぞれ異なるブロックは1つのエントリラベルにしかアタッチできません。

```haskell
        reachableFrom = IntMap.unionWith IntSet.union (IntMap.fromSet IntSet.singleton entries) strictReachableFrom
        singlyReached = flipEdges $ IntMap.filter (\ r -> IntSet.size r == 1) $ IntMap.map (IntSet.intersection entries) reachableFrom
```

エントリの一部のサブセットは、そのエントリを介してのみ到達可能なラベルのセットに関連付けられています。
これらを対応するブロックにマッピングすると、それらが互いに素であるという特性が維持されます。

さらに、この `Multiple`ブロックの中に現れることが許されているラベルだけがこの後も残ります。
後のブロックに既に割り当てられているラベルは、このラベルに重複することはありません。そのため、制御が後のコピーまで継続するようにコードを生成する必要があります。

```haskell
        handledEntries = IntMap.map (\ within -> blocks `restrictKeys` within) singlyReached
```

エントリラベルの1つが別のラベルに到達する可能性がある場合、後者をこの `Multiple`ブロックで処理することはできません。なぜなら、一方から他方へ制御フローを作る方法がないからです。
これらの未処理エントリは後続のブロックで処理する必要があります。

```haskell
        unhandledEntries = entries `IntSet.difference` IntMap.keysSet handledEntries
```

ただし、処理しているエントリポイントからしか到達できないラベルはすべて、この `Multiple`ブロック内のどこかに配置されます。
残されたラベルはこのブロックの後のどこかに配置されます。

```haskell
        handledBlocks = IntMap.unions (IntMap.elems handledEntries)
        followBlocks = blocks `IntMap.difference` handledBlocks
```

このブロックの後のブロックには、このブロックの未処理のエントリごとにエントリポイントがあり、さらに、この `Multiple`ブロックを離れるブランチごとに1つのエントリポイントがあります。

```haskell
        followEntries = unhandledEntries `IntSet.union` outEdges handledBlocks
```

最後に、エントリとラベルをこの `Multiple`ブロックの内側にあるものとそれに続くものに分割しました。
処理された各エントリポイントで再帰します。

```haskell
        makeHandler entry blocks' = relooper (IntSet.singleton entry) blocks'
        allHandlers = IntMap.mapWithKey makeHandler handledEntries
```

この時点で、すべてのハンドラを `Multiple`ブロックに入れて、` unhandled`部分を空のままにすることができます。
しかし、それは必要以上に複雑で、時にすべてのエントリラベルのハンドラを持っている場合には間違っているコードを生成します。
その場合、制御が最後のハンドラの保護に達すると、条件は常に真に評価されなければならないので、最後の `else if`文を無条件の` else`に置き換えることができます。

これを証明するには、current-block変数が現時点で持つことができる値のセットに関する正確な知識を使用します。
しかし、それを証明できるコンパイラはほとんどありません。一般的なケースでは、正確な値セットを追跡するのは困難であり、コンパイラを書く人はその努力を価値があるとは通常考えていません。

その結果、このブロックが関数の最後のものであり、すべてのハンドラが値を返すことになっている場合、ある値がすべてのパスに返されることを検証するコンパイラは、あるパスに `return`ステートメントがないと結論付けます。その経路に到達できないことはわかっていますが。

したがって、すべてのエントリポイントにハンドラがある場合は、このブロックのelse分岐になるように1つ選択します。
そうでなければ、else分岐はありません。

```haskell
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
```


不要な複数エントリブロックを排除
---------------------------------------------

```haskell
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
```

`Simple`ブロックの直後に` Multiple`ブロックが続く場合、いくつかの有用な事実がすぐにわかります：

 -  `Simple`ブロックは条件付き分岐で終わります。両方のターゲットは異なる` GoTo`ラベルです。
そうでなければ、次のブロックは `Multiple`ブロックになるのに十分なエントリポイントを持たないでしょう。

 - 条件付き分岐の各ターゲットは `Multiple`ブロックから置き換えることができるハンドラを持っているか、未処理のブロックと置き換えることができます。

 -  `Multiple`ブロックのすべての空でない分岐はこのプロセスで使われるので、コードは失われません。

 - この単純化によってコードが重複することはありません。

ここで一つ注意が必要なのは、状況によっては、いくつかのブランチで `mkGoto`ステートメントが出ていることを確認する必要があることです。
便利なことに、ここで私たちは無条件に `GoTo`ブランチで終わる空の` Simple`ブロックを挿入し、後で `structureCFG`に実際のコードの発行が必要かどうかを決定させることができます。

```haskell
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
```


最終構造化コードの生成
--------------------------------

すべての予備的な分析が邪魔にならないので、制御フローグラフをループと `if`ステートメントでいっぱいの構造化プログラムに戻す準備ができました。

このモジュールは言語固有ではないので、呼び出し側は `break`、` continue`、loop、そして `if`ステートメントを構築するための関数を提供する必要があります。
ループ関連コンストラクタはラベルを取得し、そこからループ名を生成して、マルチレベル出口をサポートします。

```haskell
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
```
