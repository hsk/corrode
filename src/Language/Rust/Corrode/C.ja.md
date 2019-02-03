これは "Literate Haskell" ソースファイルです。
ドキュメントとして読むことを意図していますが、それはコンパイル可能なソースコードでもあります。
テキストは Markdown を使用してフォーマットされているので、 GitHub はそれをうまくレンダリングできます。

このモジュールは、Cで書かれたソースコードをRustで書かれたソースコードに変換します。
まあ、それは小さなうそです。
CとRustの両方のソースについて、生のソーステキストではなく抽象表現を使用します。
C表現は [language-c](http://hackage.haskell.org/package/language-c) パッケージに由来します。
Rust表現はCorrodeの `Language.Rust.AST`モジュールにあります。

私はHaskell2010言語の標準機能に固執しようとしましたが、GHCの `ViewPatterns`拡張はこの種のコードにはあまりにも便利です。

```haskell
{-# LANGUAGE ViewPatterns #-}
```

このモジュールは、Cの「翻訳単位」（C標準からの用語で、言語-cで再利用される）をRustの「アイテム」（Rust言語仕様からの用語）のコレクションに変換する単一の関数をエクスポートします。
CとRustのASTに加えて、その他の便利なデータ構造と制御フローの抽象化をいくつかインポートします。

```haskell
module Language.Rust.Corrode.C (interpretTranslationUnit) where

import Control.Monad
import Control.Monad.ST
import Control.Monad.Trans.Class
import Control.Monad.Trans.Except
import Control.Monad.Trans.RWS.Strict
import Data.Foldable
import qualified Data.Map.Lazy as Map
import qualified Data.IntMap.Strict as IntMap
import Data.Maybe
import Data.List
import Data.STRef
import qualified Data.Set as Set
import Language.C
import Language.C.Data.Ident
import qualified Language.Rust.AST as Rust
import Language.Rust.Corrode.CFG
import Language.Rust.Corrode.CrateMap
import Text.PrettyPrint.HughesPJClass hiding (Pretty)
```

この変換は構文指向の方法で行われます。
つまり、Cの抽象構文ツリーをたどっていき、見つかった構文のそれぞれについて、同等のRustを定義する必要があります。

このようなツールは、純粋に構文指向のアプローチよりも洗練された戦略を使用することができます。
制御フローグラフなど、最近のほとんどのコンパイラが行うような中間データ構造を構築し、それらのデータ構造をファンシー分析パスで渡すことができます。

ただし、純粋に構文指向の変換には2つの利点があります。

1. 実装が簡単です。
2. そして、Corrodeの設計目標の1つであるソースプログラムからできるだけ多くの構造を保存する方が簡単です。

これまでのところ、私たちはもっと洗練されたアプローチを必要としていません、そして、できるだけ長い間構文指向のスタイルを使い続けます。

そうは言っても、変換を実行するときには、中間結果を含むデータ構造を維持する必要があります。


中間データ構造
============================

C言語は、コンパイラが翻訳単位を通過する単一の上から下へのパスでコードを生成できるように定義されています。
つまり、名前の使用はすべて、その名前が最初に宣言されている翻訳単位内のポイントの後に来る必要があります。

環境を「状態モナド」に保ち、「作家モナド」を使用して翻訳されたRustアイテムを発行することができるため、これは便利です。
これらはいくつかの簡単な操作を持つオブジェクトのための派手な用語です。

- 状態モナドは現在の環境を問い合わせたり更新したりするために使う `get`、` put`、そして `modify`操作を持っています。
- writer モナドは新しいRustアイテムを出力に追加するために使う `tell`操作を持っています。

実際には、リーダー、ライター、およびステートモナドの操作をすべて1つにまとめた複合型の `RWS`を使用します。

あなたはおそらくこれらを使うためだけにそこにあるひどいモナドチュートリアルを読む必要はありません！重要な点は、この型の別名、 `EnvMonad`があることです。これは、このモジュールを通して見ることができます。
それは環境と出力にアクセスできるコードの断片をマークします。

```haskell
type EnvMonad s = ExceptT String (RWST FunctionContext Output (EnvState s) (ST s))
```

実際、モナド操作をいくつかのヘルパー関数にまとめて、それから他の場所でのみそれらのヘルパーを使用します。
しかし、最初に、州と作家のタイプを定義しましょう。


コンテキスト
------------

```haskell
data FunctionContext = FunctionContext
    { functionReturnType :: Maybe CType
    , functionName :: Maybe String
    , itemRewrites :: ItemRewrites
    }
```


出力
----

Cのソースをたどっていくうちに、最終的な出力に含めたいRustの定義を蓄積します。
この型は、パースツリーのさらに下からバブルアップしたいデータを単に保持するだけです。
例えば、 `struct`は関数の中のループの中の宣言の中で定義されるかもしれません。
私たちはその構造をRustの中のそれ自身の項目に取り出す必要があります。

```haskell
data Output = Output
    { outputItems :: [Rust.Item]
    , outputExterns :: Map.Map String Rust.ExternItem
    , outputIncomplete :: Set.Set String
    }
```

C構文の各部分を変換して出力を生成するので、異なる部分からの出力を組み合わせる方法が必要です。
このアイデアをカプセル化した [`Monoid`](https://hackage.haskell.org/package/base-4.9.0.0/docs/Data-Monoid.html) という標準のHaskell型クラス（これはRust特性のようなものです）があります。出力を結合する

```haskell
instance Monoid Output where
```

出力をモノイドにするには、次のように指定する必要があります。

- 空の出力はどのように見えるか（これはすべてのフィールドが空のところに出力されるだけです）、

    ```haskell
        mempty = Output
            { outputItems = mempty
            , outputExterns = mempty
            , outputIncomplete = mempty
            }
    ```

-  2つの異なる出力を組み合わせる方法（これは各フィールドをペアで組み合わせることによって行います）。

    ```haskell
        mappend a b = Output
            { outputItems = outputItems a `mappend` outputItems b
            , outputExterns = outputExterns a `mappend` outputExterns b
            , outputIncomplete = outputIncomplete a `mappend` outputIncomplete b
            }
    ```

`emitItems` は Rust アイテムのリストを出力に追加します。

```haskell
emitItems :: [Rust.Item] -> EnvMonad s ()
emitItems items = lift $ tell mempty { outputItems = items }
```

`emitIncomplete` は不完全型を見たことを記録します。
私たちの目的のためには、不完全型は変換できない定義を持つものですが、誰もそれらのポインタを間接参照しない限り、この型の値へのポインタを許可することができます。

```haskell
emitIncomplete :: ItemKind -> Ident -> EnvMonad s CType
emitIncomplete kind ident = do
    rewrites <- lift (asks itemRewrites)
    unless (Map.member (kind, identToString ident) rewrites) $
        lift $ tell mempty { outputIncomplete = Set.singleton (identToString ident) }
    return (IsIncomplete ident)
```

多くの場合、型は不完全なものになるのは、それが前方宣言をしているからという理由だけです。その場合、実際の宣言が見られたら、それを不完全と見なすべきではありません。
ただし、新しく完成したタイプを反映するように環境を書き換えないことを選択しました。
代わりに、何かを調べたときに、それが不完全な型を持っていたことを発見したら、その型がそれ以降に完成したかどうかを再チェックします。
ただし、呼び出し側はまだ不完全型を処理する準備をする必要があります。

```haskell
completeType :: CType -> EnvMonad s CType
completeType orig@(IsIncomplete ident) = do
    mty <- getTagIdent ident
    fromMaybe (return orig) mty
completeType ty = return ty
```


グローバルな状態
----------------

writerモナドで使われている上記の `Output` 型は、ASTの葉から上に向かってボトムアップで累積します。
たとえば、特定の式を翻訳するときに、その部分式からの出力を確認することはできますが、隣接する式、ステートメント、または関数からの出力を調べることはできません。

これとは対照的に、ソースコードの早い段階で任意の場所から計算した情報を調べる必要がある場合があります。
そのために、後で必要になる可能性があるものをすべて保存するために、ステートモナドを使用します。

状態を2つのカテゴリに分けましょう。「グローバル」状態と「スコープ制限」状態と呼びます。
以下に説明するスコープ限定状態への変更は、これらの変更の原因となったCスコープを離れるたびに元に戻されます。
一方、グローバルな状態は永続的です。

生成したいRustパターンにはCが必要としなかった場所の名前が必要なので、時々、一意の名前を作成する必要があります。
標準的な慣用句に従って、新しい名前が必要になるたびに一意の番号を生成します。

```haskell
data GlobalState = GlobalState
    { unique :: Int
    , usedForwardRefs :: Set.Set Ident
    }
```

`uniqueName`は与えられた基数と新しいユニークな番号で新しい名前を生成します。

```haskell
uniqueName :: String -> EnvMonad s String
uniqueName base = modifyGlobal $ \ st ->
    (st { unique = unique st + 1 }, base ++ show (unique st))
```

前方宣言が参照されているが、後の定義がまだ見られていない場合は、最終的にそのシンボルが必要になるという事実を追跡する必要があるため、最終的に確認したら強制的に翻訳するようにします。

```haskell
useForwardRef :: Ident -> EnvMonad s ()
useForwardRef ident = modifyGlobal $ \ st ->
    (st { usedForwardRefs = Set.insert ident (usedForwardRefs st) }, ())
```


スコープ限定状態
-------------------

CをRustに正しく変換する上での最大の課題の1つは、Cがどの式を表すのかを知る必要があるということです。そのため、Rustの同等物が確実に同じ型を使用できるようになります。
そのためには、現在有効範囲にあるすべての名前とその型情報を追跡する必要があります。

Cには、変数、関数、および型の名前に3つの名前空間があります。

-  `struct`または` union`定義の中に導入されたフィールド名。
- 宣言によって導入される識別子。これは変数、関数、またはtypedefかもしれません。
- タグと、 `struct <tag>`、 `union <tag>`、または `enum <tag>`（C99セクション6.7.2.3）で導入されました。

例えば、以下は3つの方法すべてで `foo`という名前を使用した、Cソースの正当な断片です。

```c
typedef struct foo { struct foo *foo; } foo;
```

スコープ内で、同じネームスペースにある名前は同じものを参照する必要があります。
たとえば、同じスコープ内で `struct x`と` enum x`の両方を宣言することはできません。両者は同じタグを使用するからです。
同様に、typedefと変数を同じスコープ内に同じ名前で宣言することはできません。どちらも識別子を宣言するためです。

さびは名前空間を別の方法で分割します。
Rustでは、型は1つの名前空間にあります -  Cのように `struct`と` enum`型だけでなく、型の別名です。
変数と関数は別の名前空間にあります。
（ユニットとタプルのような構造体は実際には_両方の名前空間にありますが、幸いCorrodeはこれらの種類の構造体を生成しません。）

幸いなことに、language-cは、同じように見える名前のこれらの異なる使用法を明確にするという大変な作業をしています。
後で見るように、識別子が発生するときはいつでも、それが現れたASTコンテキストからそれにどの名前空間を使うべきかを伝えることができます。

このモジュールの最後で、C型の内部表現に戻ります。

プログラム内の任意の時点で表示される識別子、typedef、およびタグは、Cのスコープ規則によって制限されています。
それらを1つのタイプにまとめます。
この型は上で定義された `GlobalState`値への参照も運びます。

```haskell
data EnvState s = EnvState
    { symbolEnvironment :: [(Ident, EnvMonad s Result)]
    , typedefEnvironment :: [(Ident, EnvMonad s IntermediateType)]
    , tagEnvironment :: [(Ident, EnvMonad s CType)]
    , globalState :: GlobalState
    }
```

`modifyGlobal`は指定された変換に従ってグローバル状態を更新し、その変換によって計算された値を返します。

```haskell
modifyGlobal :: (GlobalState -> (GlobalState, a)) -> EnvMonad s a
modifyGlobal f = lift $ do
    st <- get
    let (global', a) = f (globalState st)
    put st { globalState = global' }
    return a
```

いくつかの名前はCで特別、Rustで特別、またはその両方です。
遭遇したときに名前を変更します。
現時点では、 `main`だけがこの特別な扱いを受けます。

```haskell
applyRenames :: Ident -> String
applyRenames ident = case identToString ident of
    "final" -> "final_"
    "fn" -> "fn_"
    "in" -> "in_"
    "let" -> "let_"
    "main" -> "_c_main"
    "match" -> "match_"
    "mod" -> "mod_"
    "proc" -> "proc_"
    "type" -> "type_"
    "where" -> "where_"
    name -> name
```

`get*Ident` は環境内の適切な名前空間から名前を検索し、それに対して持っている型情報を返します。その名前の宣言をまだ見ていない場合は` Nothing`を返します。

```haskell
getSymbolIdent :: Ident -> EnvMonad s (Maybe Result)
getSymbolIdent ident = do
    env <- lift get
    case lookup ident (symbolEnvironment env) of
        Just symbol -> fmap Just symbol
        Nothing -> case identToString ident of
            "__func__" -> getFunctionName ""
            "__FUNCTION__" -> getFunctionName ""
            "__PRETTY_FUNCTION__" -> getFunctionName "top level"
            name -> return $ lookup name builtinSymbols
    where
    getFunctionName def = do
        name <- lift (asks functionName)
        let name' = fromMaybe def name
        return $ Just Result
            { resultType = IsArray Rust.Immutable (length name' + 1) charType
            , resultMutable = Rust.Immutable
            , result = Rust.Deref (Rust.Lit (Rust.LitByteStr (name' ++ "\NUL")))
            }
    builtinSymbols =
        [ ("__builtin_bswap" ++ show w, Result
            { resultType = IsFunc (IsInt Unsigned (BitWidth w))
                [(Nothing, IsInt Unsigned (BitWidth w))] False
            , resultMutable = Rust.Immutable
            , result = Rust.Path (Rust.PathSegments ["u" ++ show w, "swap_bytes"])
            })
        | w <- [16, 32, 64]
        ]
        ++
        [ ("__FILE__", Result
            { resultType = IsPtr Rust.Immutable charType
            , resultMutable = Rust.Immutable
            , result = Rust.MethodCall (
                    Rust.Call (Rust.Var (Rust.VarName "file!")) []
                ) (Rust.VarName "as_ptr") []
            })
        , ("__LINE__", Result
            { resultType = IsInt Unsigned (BitWidth 32)
            , resultMutable = Rust.Immutable
            , result = Rust.Call (Rust.Var (Rust.VarName "line!")) []
            })
        ]

getTypedefIdent :: Ident -> EnvMonad s (String, Maybe (EnvMonad s IntermediateType))
getTypedefIdent ident = lift $ do
    env <- gets typedefEnvironment
    return (identToString ident, lookup ident env)

getTagIdent :: Ident -> EnvMonad s (Maybe (EnvMonad s CType))
getTagIdent ident = lift $ do
    env <- gets tagEnvironment
    return $ lookup ident env
```

`add * Ident`は型情報を環境に保存します。

```haskell
addSymbolIdent :: Ident -> (Rust.Mutable, CType) -> EnvMonad s String
addSymbolIdent ident (mut, ty) = do
    let name = applyRenames ident
    addSymbolIdentAction ident $ return Result
        { resultType = ty
        , resultMutable = mut
        , result = Rust.Path (Rust.PathSegments [name])
        }
    return name

addSymbolIdentAction :: Ident -> EnvMonad s Result -> EnvMonad s ()
addSymbolIdentAction ident action = lift $ do
    modify $ \ st -> st { symbolEnvironment = (ident, action) : symbolEnvironment st }

addTypedefIdent :: Ident -> EnvMonad s IntermediateType -> EnvMonad s ()
addTypedefIdent ident ty = lift $ do
    modify $ \ st -> st { typedefEnvironment = (ident, ty) : typedefEnvironment st }

addTagIdent :: Ident -> EnvMonad s CType -> EnvMonad s ()
addTagIdent ident ty = lift $ do
    modify $ \ st -> st { tagEnvironment = (ident, ty) : tagEnvironment st }
```

`addExternIdent` は `addSymbolIdent` のように型情報を環境に保存します。
しかし、外部宣言はすぐには出力に追加されません。
ヘッダーファイル内のほとんどの宣言は、どの翻訳単位でも使用されないため、出力に含めると混乱することがあります。宣言が使用されていても、この翻訳単位は同じシンボルの非 `extern` 定義を持つことができます。その場合は、完全な定義のみを発行するべきです。
そのため、この関数は、必要であることが判明した場合にどのextern宣言が発行されるかを記録します。

```haskell
addExternIdent
    :: Ident
    -> EnvMonad s IntermediateType
    -> (String -> (Rust.Mutable, CType) -> Rust.ExternItem)
    -> EnvMonad s ()
addExternIdent ident deferred mkItem = do
    action <- runOnce $ do
        itype <- deferred
        rewrites <- lift $ asks itemRewrites
        path <- case Map.lookup (Symbol, identToString ident) rewrites of
            Just renamed -> return ("" : renamed)
            Nothing -> do
                let name = applyRenames ident
                let ty = (typeMutable itype, typeRep itype)
                lift $ tell mempty { outputExterns = Map.singleton name (mkItem name ty) }
                return [name]
        return (typeToResult itype (Rust.Path (Rust.PathSegments path)))
    addSymbolIdentAction ident action
```


エラーを報告する
=================

language-cが楽しく解析するCソースコードが常にあるでしょうが、CorrodeはRustに翻訳することができません、そして、それが起こるときできるだけ私たちは役に立つ説明を報告するべきです。

```haskell
noTranslation :: (Pretty node, Pos node) => node -> String -> EnvMonad s a
noTranslation node msg = throwE $ concat
    [ show (posOf node)
    , ": "
    , msg
    , ":\n"
    , render (nest 4 (pretty node))
    ]
```

場合によっては、与えられた入力を翻訳できるはずですが、翻訳はまだ実装されていません。

```haskell
unimplemented :: (Pretty node, Pos node) => node -> EnvMonad s a
unimplemented node = noTranslation node "Corrode doesn't handle this yet"
```

一部のCソースは、C標準によれば違法ですが、それでも構文的には有効です。
Corrodeはこれらすべてのケースを検出するとは約束していませんが、そのようなエラーを検出したときに `badSource`を呼び出すことができます。

```haskell
badSource :: (Pretty node, Pos node) => node -> String -> EnvMonad s a
badSource node msg = noTranslation node
    ("illegal " ++ msg ++ "; check whether a real C compiler accepts this")
```


トップレベルの翻訳
=====================

構文指向のアプローチを取っているので、C ASTの根本である「翻訳単位」をどのように処理するかを見てみましょう。
これは単一のソースファイル（ `*.c`）に対するCの用語ですが、このコードは既に前処理されたソースで動作するので、翻訳単位にはヘッダファイルから`＃include`されたコードも含まれます。

（マクロと条件付きでコンパイルされたコードをRustの同様の構造体に変換できるといいのですが、Cプリプロセッサでまだ実行されていないソースコード用の妥当なCパーサーを書くのははるかに困難です。
前処理されたCでさえも解析するのは難しいので、それは何かを言っています。
そのため、language-cはあなたが渡したソースファイルを解析する前に `gcc` か `clang` からプリプロセッサを実行します。
ドライバコードについてはCorrodeの `Main.hs`を参照してください。）

`interpretTranslationUnit`は翻訳単位の言語CのAST表現を取り、Rust ASTのトップレベル宣言項目のリストを返します。

> **TODO**：正確にどの宣言がこのモジュールから公開されるべきかのリストとして `thisModule :: ModuleMap`を使用してください。
> このようにして `static`シンボルを公開し、` `static` 'シンボルを隠すことができます。
> この翻訳単位で定義されていないインクルードシンボルについては、それらの `extern`宣言をpublicにしてください。
> 指定されたとおりに定義の名前を変更しますが、元の名前はexternsのための `＃[link_name =" ... "]`とトップレベルの定義のための `＃[export_name =" ... "]`に保存します。

```haskell
interpretTranslationUnit :: ModuleMap -> ItemRewrites -> CTranslUnit -> Either String [Rust.Item]
interpretTranslationUnit _thisModule rewrites (CTranslUnit decls _) = case err of
    Left msg -> Left msg
    Right _ -> Right items'
    where
```

ここでは、翻訳単位の最上位レベルで、前述の作家と州のモナドから翻訳結果を抽出する必要があります。
具体的には、

- 環境を最初は空に設定します。
- 翻訳が完了したら、中間型情報は必要ないので、最終的な環境を破棄します。
- そして翻訳中に発行されたアイテムとエクステンダの最終リストを取得します。

```haskell
    initFlow = FunctionContext
        { functionReturnType = Nothing
        , functionName = Nothing
        , itemRewrites = rewrites
        }
    initState = EnvState
        { symbolEnvironment = []
        , typedefEnvironment = []
        , tagEnvironment = []
        , globalState = GlobalState
            { unique = 1
            , usedForwardRefs = Set.empty
            }
        }
    (err, output) = runST (evalRWST (runExceptT (mapM_ perDecl decls)) initFlow initState)
```

初期環境が設定されたら、次のレベルの抽象構文ツリーに降りることができます。ここで、language-c型は、宣言ごとに3つの可能な種類があることを示しています。

```haskell
    perDecl (CFDefExt f) = interpretFunction f
    perDecl (CDeclExt decl') = do
        binds <- interpretDeclarations makeStaticBinding decl'
        emitItems binds
    perDecl decl = unimplemented decl
```

不完全な型（上の `emitIncomplete`を参照）は早く参照され、そして後で完了されるかもしれません。
その場合は、完全な定義のみを発行する必要があります。

しかし、それらが完全に完成していない場合は、参照によってしか受け渡しできないRustタイプをいくつか発行する必要があります。
Rustコンパイラが不完全型の値を構築、コピー、または消費することを許可することはできません。
また、それぞれの不完全型を他のすべての不完全型と区別したいと考えています。

それぞれの不完全な型に対して、プライベートな `enum`型を作成することによって、これらの要件を満たしています。
私たちは型にコンストラクタを与えないので、新しい値を構築することはできず、それを `match`することもできません。
他の型の翻訳とは異なり、このenumの `repr`を宣言していませんし、それに対して` Copy`や `Clone`を派生させていません。

```haskell
    completeTypes = Set.fromList $ catMaybes
        [ case item of
            Rust.Item _ _ (Rust.Struct name _) -> Just name
            _ -> Nothing
        | item <- outputItems output
        ]
    incompleteTypes = outputIncomplete output `Set.difference` completeTypes
    incompleteItems =
        [ Rust.Item [] Rust.Private (Rust.Enum name [])
        | name <- Set.toList incompleteTypes
        ]
```

次に、外部宣言のリストからローカルに定義された名前を削除します。
宣言に遭遇したときに、何かが実際に外部的なものであるかどうかを判断することはできませんが、この翻訳単位のすべての記号を収集したら、それは明らかになります。

```haskell
    itemNames = catMaybes
        [ case item of
            Rust.Item _ _ (Rust.Function _ name _ _ _) -> Just name
            Rust.Item _ _ (Rust.Static _ (Rust.VarName name) _ _) -> Just name
            _ -> Nothing
        | item <- outputItems output
        ]

    externs' =
        [ extern
        | (name, extern) <- Map.toList (outputExterns output)
        , name `notElem` itemNames
        ]
```

型宣言が他の宣言で実際に使用されていない場合は、それを削除します。
Cヘッダは多くの型を宣言します、そしてそれらのほとんどは与えられたどのソースファイルでも使われません。

これは単なる出力のクリーンアップではありません。
また、実際に使用されていない限り、まだ翻訳できないタイプを含むヘッダーを翻訳することもできます。

```haskell
    items = incompleteItems ++ outputItems output
```

フィルタリング後に外部宣言がある場合は、それらを `extern { }` ブロックで囲む必要があります。
慣例により、他の項目の前に配置します。

```haskell
    items' = if null externs'
        then items
        else Rust.Item [] Rust.Private (Rust.Extern externs') : items
```


宣言
============

C言語の宣言は、トップレベルでも関数に対してローカルでも、構文的には同じように見えます。
`interpretDeclarations`は両方の場合を扱います。

しかし、Rustには別の一連のケースがあり、それらを次のようにマップする必要があります。一方ローカル変数は `let`束縛文を使って宣言されます。

そのため `interpretDeclarations`は関数` makeBinding`によってパラメータ化されています。これは非静的C変数宣言から `static`アイテムまたは` let`バインディングを構築することができます。
どちらの場合も、バインディングは

 - 可変でも不変でもよい (`const`)。
 - 変数名を持ちます。
 - 型を持つ（型が一般に推論される可能性がある「let」束縛に対してさえも明示的になることを選択する）。
 - そして初期化式があるかもしれません。

関数の戻り型は、それがどの種類の束縛を構成しているかによって異なります。

```haskell
type MakeBinding s a = (Rust.ItemKind -> a, Rust.Mutable -> Rust.Var -> CType -> NodeInfo -> Maybe CInit -> EnvMonad s a)
```

`makeBinding` の適切な実装を提供するのは呼び出し側次第ですが、他に便利に使用するためにここで定義する2つの賢明な選択しかありません。

> ** FIXME **：スカラー以外の静的変数に正しいデフォルト値を構築します。

```haskell
makeStaticBinding :: MakeBinding s Rust.Item
makeStaticBinding = (Rust.Item [] Rust.Private, makeBinding)
    where
    makeBinding mut var ty node minit = do
        expr <- interpretInitializer ty (fromMaybe (CInitList [] node) minit)
        return $ Rust.Item attrs Rust.Public
            (Rust.Static mut var (toRustType ty) expr)
    attrs = [Rust.Attribute "no_mangle"]

makeLetBinding :: MakeBinding s Rust.Stmt
makeLetBinding = (Rust.StmtItem [], makeBinding)
    where
    makeBinding mut var ty _ minit = do
        mexpr <- mapM (interpretInitializer ty) minit
        return $ Rust.Let mut var (Just (toRustType ty)) mexpr
```

変数宣言の変換方法がわかったので、宣言の場所に関係なく、他のすべては同じように機能します。
この時点で処理しなければならない最も単純なケースの例を示します。

```c
// 変数定義、初期値
int x = 42;

// 変数宣言、他のモジュールを参照している可能性があります
extern int y;

// エイリアスタイプ
typedef int my_int;

// 構造体定義
struct my_struct {
    //型エイリアスを使う
    my_int field;
};

// 以前に定義された構造体を使用した変数定義
struct my_struct s;

// 関数プロトタイプ。他のモジュールを参照している可能性があります
int f(struct my_struct *);
extern int g(my_int i);

// 確実にローカルな関数の関数プロトタイプ
static int h(void);
```

注意すべきいくつかの重要事項：

- Cは、宣言内で**zero** "declarators"のように少ない数を許可します。
上記の例では、 `struct my_struct`の定義は0個の宣言子を持ちますが、他のすべての宣言はそれぞれ1個の宣言子を持ちます。

- `struct`/`union`/`enum` の定義は一種の宣言指定子です、例えば `int` や `const` や `static` のように、宣言の中からそれらを探し出す必要があります。他の `struct`定義、そしておそらく新しい型を持つものとして直ちに宣言された変数を使ったもの。
Rustでは、それぞれの `struct`定義は別々の項目でなければならず、その後、新しい型を使って変数を宣言することができます。

- 構文的には、 `typedef`は` static`や `extern`と同じように記憶クラス指定子です。
しかし意味的には、それらはもっと違いがあることができませんでした！

そして、ここにCの宣言構文がどれほど奇妙に複雑であるかを示す法的宣言がいくつかあります。

- 宣言指定子しかなく宣言子しかないため、これらは効果がありません。

    ```c
    int;
    const typedef volatile void;
    ```

- int型の変数と、intを返す関数の関数プロトタイプの両方を宣言します。

    ```c
    int i, fun(void);
    ```

- 型intの変数、intへのポインタ、intへのポインタを返す関数、およびintを返す関数へのポインタをそれぞれ宣言します。

    ```c
    int j, *p, *fip(void), (*fp)(void);
    ```

-  `int` と `int *` の両方の型エイリアスを宣言します。

    ```c
    typedef int new_int, *int_ptr;
    ```

それでは、 `interpretDeclarations` がこれらすべてのケースをどのように扱うかを見てみましょう。
それは `MakeBinding` 実装の一つと単一のC宣言を受け取ります。
これは `MakeBinding` を使って構築されたバインディングのリストを返し、また必要に応じて環境と出力を更新します。

```haskell
interpretDeclarations :: MakeBinding s b -> CDecl -> EnvMonad s [b]
interpretDeclarations (fromItem, makeBinding) declaration@(CDecl specs decls _) = do
```

まず、宣言指定子によって記述された型の内部表現を取得するために `baseTypeOf`を呼び出します。
その関数はまた、この宣言で定義された `struct`を環境に追加するので、それらが再度参照されればそれらを調べて、それらのstructを出力の項目として出力することができます。

宣言子がない場合、 `baseTypeOf` の副作用はこの関数からの唯一の出力です。

```haskell
    (storagespecs, baseTy) <- baseTypeOf specs
```

宣言子があれば、もしあれば、それぞれに対してどの種類の Rust 宣言を発行するかを考える必要があります。

```haskell
    mbinds <- forM decls $ \ declarator -> do
```

この関数は `language-c`が"トップレベルの宣言 "と呼ぶものに対してのみ使われ、そして"空でない `init-declarator-list`の要素は`（ちょうどdeclr、init？、Nothing）という形式です。 ）
宣言子 `declr`が存在し、抽象的でなければならず、初期化式はオプションです。"

```haskell
        (decl, minit) <- case declarator of
            (Just decl, minit, Nothing) -> return (decl, minit)
            (Nothing, _, _) -> badSource declaration "absent declarator"
            (_, _, Just _) -> badSource declaration "bitfield declarator"

        -- FIXME: if `specs` is a typedef reference, dig more derived out of that.
        (ident, derived) <- case decl of
            CDeclr (Just ident) derived _ _ _ -> return (ident, derived)
            _ -> badSource decl "abstract declarator"

        deferred <- derivedDeferredTypeOf baseTy decl []
        case (storagespecs, derived) of
```

それぞれの `typedef` 宣言子は環境に追加されます。
それらは初期化子を持ってはならず、宣言を返さないので、 `typedef`の唯一の効果は環境を更新することです。

> ** TODO **：エイリアスを基本的な型に置き換えるのではなく、可能なところならどこでもtype-alias項目を `typedef`ごとに発行して別名を使うのがいいでしょう。
このようにして、入力プログラムの構造をより多く保存します。
しかし、それは常に可能というわけではないので、これには慎重な検討が必要です。

```haskell
            (Just (CTypedef _), _) -> do
                when (isJust minit) (badSource decl "initializer on typedef")
                addTypedefIdent ident deferred
                return Nothing
```

`typedef`以外の宣言は初期化子を持つことができます。
各宣言子は新しいシンボルとして環境に追加されます。

関数定義は同じ変換単位になければならないため、静的関数プロトタイプは変換する必要はありません。
ただし、環境内で関数の型シグネチャを保持する必要があります。

```haskell
            (Just (CStatic _), CFunDeclr{} : _) -> do
                addSymbolIdentAction ident $ do
                    itype <- deferred
                    useForwardRef ident
                    return (typeToResult itype (Rust.Path (Rust.PathSegments [applyRenames ident])))
                return Nothing
```

他の関数プロトタイプは、関数定義が同じ翻訳単位に現れない限り翻訳する必要があります。それを実行し、後で重複を整理します。

```haskell
            (_, CFunDeclr{} : _) -> do
                addExternIdent ident deferred $ \ name (_mut, ty) -> case ty of
                    IsFunc retTy args variadic ->
                        let formals =
                                [ (Rust.VarName argName, toRustType argTy)
                                | (idx, (mname, argTy)) <- zip [1 :: Int ..] args
                                , let argName = maybe ("arg" ++ show idx) (applyRenames . snd) mname
                                ]
                        in Rust.ExternFn name formals variadic (toRustRetType retTy)
                    _ -> error (show ident ++ " is both a function and not a function?")
                return Nothing
```

同一の非外部宣言が同一の翻訳単位内に現れない限り、非機能外部は翻訳される必要があります。それを実行し、後で重複を整理します。

```haskell
            (Just (CExtern _), _) -> do
                addExternIdent ident deferred $ \ name (mut, ty) ->
                    Rust.ExternStatic mut (Rust.VarName name) (toRustType ty)
                return Nothing
```

ストレージクラス `static`を持つ宣言は常にRust静的アイテムを構築する必要があります。
これらの項目は、それが単なるゼロ当量の初期化子であっても、常に初期化子を持ちます。
アイテム（実際には `ItemKind`）を他のバインディングに返すのと同じ型に変えるために呼び出し側の` fromItem`コールバックを使います。

```haskell
            (Just (CStatic _), _) -> do
                IntermediateType
                    { typeMutable = mut
                    , typeRep = ty } <- deferred
                name <- addSymbolIdent ident (mut, ty)
                expr <- interpretInitializer ty (fromMaybe (CInitList [] (nodeInfo decl)) minit)
                return (Just (fromItem
                    (Rust.Static mut (Rust.VarName name) (toRustType ty) expr)))
```

それ以外は変換する変数宣言です。
これが `makeBinding`を使う唯一のケースです。
イニシャライザがある場合は、それも翻訳する必要があります。下記参照。

```haskell
            _ -> do
                IntermediateType
                    { typeMutable = mut
                    , typeRep = ty } <- deferred
                name <- addSymbolIdent ident (mut, ty)
                binding <- makeBinding mut (Rust.VarName name) ty (nodeInfo decl) minit
                return (Just binding)
```

`Nothing` を返さなかった宣言子に対して生成されたバインディングを返します。

```haskell
    return (catMaybes mbinds)
```

```haskell
interpretDeclarations _ node@(CStaticAssert {}) = unimplemented node
```

初期化
==============

C99の6.7.8節で説明されている初期化の一般的な形式は、初期化子構造を含みます。
C初期化子を変換するために公開したいインターフェースは非常に単純です。初期化しようとしているものの型とC初期化子を考えれば、初期化されるはずのC式に対応するRust式を生成します。

```haskell
interpretInitializer :: CType -> CInit -> EnvMonad s Rust.Expr
```

残念ながら、必要なものがすべて揃うようになるまで、このセクションの最後までこの関数の実際の実装を遅らせる必要があります。

問題は、Cでは、同じ初期化を表現する方法が多すぎることです。
たとえば、次のような構造体の定義を取ります

```c
struct Foo { struct Bar b; int z; }
struct Bar { int x; int y; }
```

それで、以下はすべて等価です

```c
struct Foo s = { 1, 2, 3 }
struct Foo s = { .b = { .x = 1, .y = 2 }, .z = 3 }
struct Foo s = { .b.y = 1, .b = { 1, 2 }, 3 }
struct Foo s = { (struct Bar) { 1, 2 }, 3 }
```

初期化式を操作して構成するための標準的な形式が必要です。
それから、C初期化式をいくつかのステップで扱うことができます。それらを標準形式に変換することから始め、それに応じてそれらを一緒に構成し、最後にそれらをRust式に変換します。

私たちの標準形は基本式を持つことができ、それは（私たちが集合体を初期化しているなら）そのフィールドのいくつかが上書きされるかもしれません。
基本式が存在する場合は、このオブジェクトの以前のすべての初期化子をオーバーライドします。
そうでなければ、 `IntMap`で指定されていないすべてのフィールドはそれらのゼロと等価な値に初期化されます。

```haskell
data Initializer
    = Initializer (Maybe Rust.Expr) (IntMap.IntMap Initializer)
```

```haskell
scalar :: Rust.Expr -> Initializer
scalar expr = Initializer (Just expr) IntMap.empty
```

イニシャライザを組み合わせることは連想バイナリ演算であることに注意してください。
これは、2つのイニシャライザを組み合わせるための操作を表すために、再びMonoid型クラスを使う動機となります。

```haskell
instance Monoid Initializer where
```
- この場合のidentity要素は空のイニシャライザになります。
これは、それが別の初期化子と組み合わされると（左または右から）、結果は他の初期化子になるからです。

    ```haskell
        mempty = Initializer Nothing IntMap.empty
    ```

 -  2つの初期化子を組み合わせるとき、右側のものは左側のものによってなされた定義を上書き/影付けします。

    ```haskell
        mappend _ b@(Initializer (Just _) _) = b
        mappend (Initializer m a) (Initializer Nothing b) =
            Initializer m (IntMap.unionWith mappend a b)
    ```

さて、最初にこれらの初期化子を構築することに私たち自身が関心を持つ必要があります。
現在のオブジェクトを追跡する必要があります（セクション6.7.8のポイント17を参照）。
「指定子」は型の中の位置を記述します。

```haskell
type CurrentObject = Maybe Designator

data Designator
  = Base CType
```

* 指している基本オブジェクトの型をエンコードする

```haskell
  | From CType Int [CType] Designator
  deriving(Show)
```

* 指しているオブジェクトのタイプ、親内のインデックス、親内の残りのフィールド、および親指定子をエンコードします。

いくつかの場所では、指定されたオブジェクトの種類を知る必要があります。

```haskell
designatorType :: Designator -> CType
designatorType (Base ty) = ty
designatorType (From ty _ _ _) = ty
```

次に、指定子のリストと現在の型を指定して、最も一般的な現在のオブジェクトを計算できます。

```haskell
objectFromDesignators :: CType -> [CDesignator] -> EnvMonad s CurrentObject
objectFromDesignators _ [] = pure Nothing
objectFromDesignators ty desigs = Just <$> go ty desigs (Base ty)
    where

    go :: CType -> [CDesignator] -> Designator -> EnvMonad s Designator
    go _ [] obj = pure obj
    go (IsArray _ size el) (CArrDesig idxExpr _ : ds) obj = do
        idx <- interpretConstExpr idxExpr
        go el ds (From el (fromInteger idx) (replicate (size - fromInteger idx - 1) el) obj)
    go (IsStruct name fields) (d@(CMemberDesig ident _) : ds) obj = do
        case span (\ (field, _) -> applyRenames ident /= field) fields of
            (_, []) -> badSource d ("designator for field not in struct " ++ name)
            (earlier, (_, ty') : rest) ->
                go ty' ds (From ty' (length earlier) (map snd rest) obj)
    go ty' (d : _) _ = badSource d ("designator for " ++ show ty')
```

しかし、イニシャライザの中には指定子を持たないものもある可能性があるので（その場合、イニシャライザは暗黙のうちに次のオブジェクトに適用されます）、現在のオブジェクトから最も一般的な次のオブジェクトを計算する方法が必要です。 tは初期化中のものの終わりに達しました。

```haskell
nextObject :: Designator -> CurrentObject
nextObject Base{} = Nothing
nextObject (From _ i (ty : remaining) base) = Just (From ty (i+1) remaining base)
nextObject (From _ _ [] base) = nextObject base
```

以下のいずれかの場合、初期化式の型は、初期化しているオブジェクトの型と互換性があります。

- 両方とも構造体型を持ち、同じ `struct`です、
- あるいはどちらも構造体型を持たない。

後者の場合、スカラー型を必要に応じて他の型にキャストできるため、それらがどの型であるかはチェックしません。

```haskell
compatibleInitializer :: CType -> CType -> Bool
compatibleInitializer (IsStruct name1 _) (IsStruct name2 _) = name1 == name2
compatibleInitializer IsStruct{} _ = False
compatibleInitializer _ IsStruct{} = False
compatibleInitializer _ _ = True
```

「最も一般的な（オブジェクト）」という表現を何度か使いました。
これは、指定子だけでは初期化される内容を正確に判断するのに十分ではないためです。型を比較す​​る必要もあります。

2つに互換性のある型がある場合、初期化式は現在のオブジェクトを初期化します。
そうでなければ、現在のオブジェクトの最初のサブオブジェクトを参照するように指示子を拡張し、_ objectが初期化子と互換性があるかどうかをチェックします。
互換性のあるオブジェクトが見つからずにサブオブジェクトがなくなるまで、最初のサブオブジェクトを降り続けます。

```haskell
nestedObject :: CType -> Designator -> Maybe Designator
nestedObject ty desig = case designatorType desig of
    IsArray _ size el -> Just (From el 0 (replicate (size - 1) el) desig)
    ty' | ty `compatibleInitializer` ty' -> Just desig
    IsStruct _ ((_ , ty') : fields) ->
        nestedObject ty (From ty' 0 (map snd fields) desig)
    _ -> Nothing
```

これらのヘルパーを与えられて、我々は今Cイニシャライザを私たちのイニシャライザに翻訳する立場にあります。

式のリストがあると、まずすべての指定子を内部表現に解析します。

```haskell
translateInitList :: CType -> CInitList -> EnvMonad s Initializer
translateInitList ty list = do

    objectsAndInitializers <- forM list $ \ (desigs, initial) -> do
        currObj <- objectFromDesignators ty desigs
        pure (currObj, initial)
```

次に、開始時の現在のオブジェクト（ `base`）を選ぶ必要があります。
集約型の場合、最初の現在のオブジェクトは最初のフィールドを指しますが、スカラー型の場合はプリミティブそれ自体を指します。
例えば

```c
struct point { int x, y };

int i = { 1, 3 };
struct point p = { 1, 3 };
```

最初の例では、iは構造体ではないので、i全体が1に初期化されます（3は無視されます）。
一方、2番目の例では、 `p`は構造体なので、` 1`と `3`に初期化されるのは` p`のフィールドです。

```haskell
    let base = case ty of
                    IsArray _ size el -> From el 0 (replicate (size - 1) el) (Base ty)
                    IsStruct _ ((_,ty'):fields) -> From ty' 0 (map snd fields) (Base ty)
                    _ -> Base ty
```

最後に、リスト全体の初期化子を計算する準備が整いました。
指定子とその初期化子のリストを現在のオブジェクトに沿って左から右へとたどります（後続の初期化子に指定子がない場合）。

最後に到達すると、初期化子リストはそれらを囲む初期化子の現在のオブジェクトに影響を与えないため、最後の現在のオブジェクトを破棄します。

```haskell
    (_, initializer) <- foldM resolveCurrentObject (Just base, mempty) objectsAndInitializers
    return initializer
```

指定子が指定されていない場合、Resolutionは現在のオブジェクトを使用します。
次に使う要素の新しい現在のオブジェクトと、この要素が初期化したオブジェクトの部分を表す上記の `Initializer`型を返します。

```haskell
resolveCurrentObject
    :: (CurrentObject, Initializer)
    -> (CurrentObject, CInit)
    -> EnvMonad s (CurrentObject, Initializer)
resolveCurrentObject (obj0, prior) (obj1, cinitial) = case obj1 `mplus` obj0 of
    Nothing -> return (Nothing, prior)
    Just obj -> do
```

提供された初期化子が別の初期化子リストである場合、初期化子は現在のオブジェクトに対するものでなければなりません。
それが単なる初期化式であれば、式を変換してそれがどんな型を持っているかを調べ、 `nestedObject`を使って初期化する対応するサブオブジェクトを見つけます。

```haskell
        (obj', initial) <- case cinitial of
            CInitList list' _ -> do
                initial <- translateInitList (designatorType obj) list'
                return (obj, initial)
            CInitExpr expr _ -> do
                expr' <- interpretExpr True expr
                case nestedObject (resultType expr') obj of
                    Nothing -> badSource cinitial "type in initializer"
                    Just obj' -> do
                        let s = castTo (designatorType obj') expr'
                        return (obj', scalar s)
```

正しい現在のオブジェクトに決め、そのための中間の `Initializer`を構築したので、前者のそれぞれの指示子に対する最小の集約イニシャライザで後者をラップする必要があります。

```haskell
        let indices = unfoldr (\o -> case o of
                                 Base{} -> Nothing
                                 From _ j _ p -> Just (j,p)) obj'
        let initializer = foldl (\a j -> Initializer Nothing (IntMap.singleton j a)) initial indices

        return (nextObject obj', prior `mappend` initializer)
```

最後に、このセクションの冒頭近くで宣言した完全な `interpretInitializer`関数を実装することができます。

初期化子リストの中で、初期化子式を適用するのに適したサブオブジェクトを検索するために上の `nestedObject`を使いました。
しかし、ここでは、初期化子リストの外では、C99の6.7.8段落13は、ただちに互換性のある型以外は何も許可しないようです。 GCC、Clang、およびICCはすべて、構造体に対するサブオブジェクト型スカラー初期化子を拒否します。

```haskell
interpretInitializer ty initial = do
    initial' <- case initial of
        CInitExpr expr _ -> do
            expr' <- interpretExpr True expr
            if resultType expr' `compatibleInitializer` ty
                then pure $ scalar (castTo ty expr')
                else badSource initial "initializer for incompatible type"
        CInitList list _ -> translateInitList ty list

    zeroed <- zeroInitialize initial' ty
    helper ty zeroed

    where
```

Cで最も単純な初期化タイプはゼロ初期化です。
ターゲットの基礎となるメモリがちょうどゼロになるように初期化されます。

```haskell
    zeroInitialize i@(Initializer Nothing initials) origTy = completeType origTy >>= \ t -> case t of
        IsBool{} -> return $ scalar (Rust.Lit (Rust.LitBool False))
        IsVoid{} -> badSource initial "initializer for void"
        IsInt{} -> return $ scalar (Rust.Lit (Rust.LitInt 0 Rust.DecRepr (toRustType t)))
        IsFloat{} -> return $ scalar (Rust.Lit (Rust.LitFloat "0" (toRustType t)))
        IsPtr{} -> return $ scalar (Rust.Cast 0 (toRustType t))
        IsArray _ size _ | IntMap.size initials == size -> return i
        IsArray _ size elTy -> do
            elInit <- zeroInitialize (Initializer Nothing IntMap.empty) elTy
            el <- helper elTy elInit
            return (Initializer (Just (Rust.RepeatArray el (fromIntegral size))) initials)
        IsFunc{} -> return $ scalar (Rust.Cast 0 (toRustType t))
        IsStruct _ fields -> do
            let fields' = IntMap.fromDistinctAscList $ zip [0..] $ map snd fields
            let missing = fields' `IntMap.difference` initials
            zeros <- mapM (zeroInitialize (Initializer Nothing IntMap.empty)) missing
            return (Initializer Nothing (IntMap.union initials zeros))
        IsEnum{} -> unimplemented initial
        IsIncomplete _ -> badSource initial "initialization of incomplete type"
    zeroInitialize i _ = return i
```

```haskell
    helper _ (Initializer (Just expr) initials) | IntMap.null initials = return expr
    helper (IsArray _ _ el) (Initializer expr initials) = case expr of
        Nothing -> Rust.ArrayExpr <$> mapM (helper el) (IntMap.elems initials)
        Just _ -> unimplemented initial
    helper strTy@(IsStruct str fields) (Initializer expr initials) =
        Rust.StructExpr str <$> fields' <*> pure expr
        where
        fields' = forM (IntMap.toList initials) $ \ (idx, value) ->
            case drop idx fields of
            (field, ty') : _ -> do
                value' <- helper ty' value
                return (field, value')
            [] -> noTranslation initial ("internal error: " ++ show strTy ++ " doesn't have enough fields to initialize field " ++ show idx)
    helper _ _ = badSource initial "initializer"
```


機能定義
====================

C関数定義は単一のRustアイテムに変換されます。

```haskell
interpretFunction :: CFunDef -> EnvMonad s ()
interpretFunction (CFunDef specs declr@(CDeclr mident _ _ _ _) argtypes body _) = do
```

この関数が `static`と宣言されているかどうかに基づいてこの関数がこのモジュールの外で見えるかどうかを決定します。

```haskell
    (storage, baseTy) <- baseTypeOf specs
    (attrs, vis) <- case storage of
        Nothing -> return ([Rust.Attribute "no_mangle"], Rust.Public)
        Just (CStatic _) -> return ([], Rust.Private)
        Just s -> badSource s "storage class specifier for function"
```

`const`は有効ですが、Cの関数の戻り値の型には意味がありません。
存在したかどうかは無視します。 生成されたRustに `mut`キーワードを入れることができる場所はありません。

```haskell
    let go name funTy = do
```

Rustはそれらをサポートしていないため、可変関数の定義は許可されていません。

```haskell
            (retTy, args) <- case funTy of
                IsFunc _ _ True -> unimplemented declr
                IsFunc retTy args False -> return (retTy, args)
                _ -> badSource declr "function definition"
```

`main`の翻訳には特別な注意が必要です。下記の `wrapMain`を参照してください。

```haskell
            when (name == "_c_main") (wrapMain declr name (map snd args))
```

return型を利用できるようにしながら、この関数の本体の新しいスコープを開いて、 `return`文を正しく翻訳できるようにします。

```haskell
            let setRetTy flow = flow
                    { functionReturnType = Just retTy
                    , functionName = Just name
                    }
            f' <- mapExceptT (local setRetTy) $ scope $ do
```

それぞれの仮パラメータを新しい環境にシンボルとして追加します。

> ** XXX **：現在すべてのパラメータが名前を持つことを要求していますが、Cが実際にそれを要求しているかどうかはわかりません。
そうでない場合は、Rustではすべての名前を付ける必要があるため、匿名パラメータにはダミー名を作成する必要がありますが、元のプログラムではパラメータにアクセスできないため、ダミー名を環境に追加しないでください。

```haskell
                formals <- sequence
                    [ case arg of
                        Just (mut, argident) -> do
                            argname <- addSymbolIdent argident (mut, ty)
                            return (mut, Rust.VarName argname, toRustType ty)
                        Nothing -> badSource declr "anonymous parameter"
                    | (arg, ty) <- args
                    ]
```

制御が関数の最後で終了した場合、メイン関数に `0`を返します（C99セクション5.1.2.2.3、"プログラムの終了 "）。
他の関数のために `return;`を見逃した `void`関数の問題を捕らえるために` return; `ステートメントを挿入します（値を返す関数のためにこのreturnステートメントは到達不可能であるため削除されるべきです）。

```haskell
                let returnValue = if name == "_c_main" then Just 0 else Nothing
                    returnStatement = Rust.Stmt (Rust.Return returnValue)
```

関数の本体を解釈してください。

```haskell
                body' <- cfgToRust declr (interpretStatement body (return ([returnStatement], Unreachable)))
```

ボディのHaskell型は `CStatement`ですが、language-cはそれが特に複合文（` CCompound`コンストラクタ）であることを保証します。
それに頼るのではなく、あらゆる種類のステートメントにすることを許可し、その結果のRustステートメントをRustブロックに強制変換するために `statementsToBlock`を使用します。

Cは最後の式が関数の結果となることを可能にするRustの構文を持っていないので、この生成されたブロックは決して最後の式を持ちません。

```haskell
                return (Rust.Item attrs vis
                    (Rust.Function [Rust.UnsafeFn, Rust.ExternABI Nothing] name formals (toRustRetType retTy)
                        (statementsToBlock body')))

            emitItems [f']
```

本体を評価する前にこの関数をグローバルに追加すると、再帰呼び出しが機能します。
（関数定義は匿名にはできません。）

```haskell
    ident <- case mident of
        Nothing -> badSource declr "anonymous function definition"
        Just ident -> return ident

    let name = applyRenames ident
    let funTy itype = typeToResult itype (Rust.Path (Rust.PathSegments [name]))
    deferred <- fmap (fmap funTy) (derivedDeferredTypeOf baseTy declr argtypes)
    alreadyUsed <- lift $ gets (usedForwardRefs . globalState)
    case vis of
        Rust.Private | ident `Set.notMember` alreadyUsed -> do
            action <- runOnce $ do
                ty <- deferred
                go name (resultType ty)
                return ty
            addSymbolIdentAction ident action
        _ -> do
            ty <- deferred
            addSymbolIdentAction ident $ return ty
            go name (resultType ty)
```


`main`の特別な場合の翻訳
==================================

Cでは、 `main`はこれらの型の1つを持つ関数であるべきです。

```c
int main(void);
int main(int argc, char *argv[]);
int main(int argc, char *argv[], char *envp[]);
```

Rustでは、 `main`は次のように宣言されている必要があります（ただし、戻り型は通常暗黙のうちに省略されるはずです）。

```rust
fn main() -> () { }
```

そのため、 `main`という名前のC関数に遭遇したとき、それをそのまま翻訳することはできません。そうしないと、Rustはそれを拒否します。
代わりに名前を変更し（上記の `applyRenames`を参照）、コマンドライン引数と環境を取得してそれらを名前を変更した` main`に渡すラッパー関数を発行します。

```haskell
wrapMain :: CDeclr -> String -> [CType] -> EnvMonad s ()
wrapMain declr realName argTypes = do
```

Cのmainが期待する引数の型に基づいて、ラッパーが何をすべきかを決定します。
異なる場合のコードは2つの部分に分けられます：いくつかの変数を `束縛 'するいくつかの設定ステートメント。そして実際の `main`関数に引数として渡される式のリスト。おそらくそれらの` let`束縛変数を使用します。

```haskell
    (setup, args) <- wrapArgv argTypes
```

本物の `main`は、私たちが翻訳するすべての関数のように、` unsafe fn`として翻訳されているので、それに対する呼び出しを `unsafe`ブロックでラップする必要があります。
そして、Rustプログラムは `main`から終了ステータスを返さないので、それがRustの` std :: process :: exit`関数に返す終了ステータスコードを渡す必要があります。

```haskell
    let ret = Rust.VarName "ret"
    emitItems [Rust.Item [] Rust.Private (
        Rust.Function [] "main" [] (Rust.TypeName "()") (statementsToBlock (
            setup ++
            [ bind Rust.Immutable ret $
                Rust.UnsafeExpr $ Rust.Block [] $ Just $
                call realName args ] ++
            exprToStatements (call "::std::process::exit" [Rust.Var ret])
        )))]
    where
```

手作業でASTコンストラクタを書くのは面倒です。
これは私たちがより短いコードを書くことを可能にするいくつかのヘルパー関数です。
それぞれがこの機能のニーズに特化しています。

-  `bind`は推論された型と初期値で` let`バインディングを作成します。

    ```haskell
        bind mut var val = Rust.Let mut var Nothing (Just val)
    ```

-  `call`は静的に知られている関数だけを呼び出せ、関数ポインタは呼び出せません。

    ```haskell
        call fn args = Rust.Call (Rust.Var (Rust.VarName fn)) args
    ```

-  `chain`はメソッド呼び出しを生成しますが、最後の引数としてオブジェクトを使うので、逆に読みます。

    ```haskell
        chain method args obj = Rust.MethodCall obj (Rust.VarName method) args
    ```

    `a（）。b（）。c（）`は次のように書かれています。

    ``` { .haskell .ignore }
    chain "c" [] $ chain "b" [] $ call "a" []
    ```

それでは引数の型を調べてみましょう。
`main`が`（void） `と宣言されていれば、引数や引数を渡す必要はありません。

```haskell
    wrapArgv [] = return ([], [])
```

しかし、それが `（int、char * argv []）`と宣言されていれば、 `std :: env :: args_os（）`を呼び出して引数文字列をCスタイルの文字列に変換する必要があります。

```haskell
    wrapArgv (argcType@(IsInt Signed (BitWidth 32))
            : IsPtr Rust.Mutable (IsPtr Rust.Mutable ty)
            : rest) | ty == charType = do
        (envSetup, envArgs) <- wrapEnvp rest
        return (setup ++ envSetup, args ++ envArgs)
        where
        argv_storage = Rust.VarName "argv_storage"
        argv = Rust.VarName "argv"
        str = Rust.VarName "str"
        vec = Rust.VarName "vec"
        setup =
```

それぞれの引数文字列をバイトのベクトルに変換し、終端のNUL文字を追加し、そのベクトルへの参照を保存して、実際の `main`が返されるまで割り当て解除されないようにします。

`OsString`を` Vec <u8> `に変換することは私たちがUnix特有の` OsStringExt`特性をスコープに持ってくる場合にのみ許されます。

```haskell
            [ Rust.StmtItem [] (Rust.Use "::std::os::unix::ffi::OsStringExt")
            , bind Rust.Mutable argv_storage $
                chain "collect::<Vec<_>>" [] $
                chain "map" [
                    Rust.Lambda [str] (Rust.BlockExpr (Rust.Block
                        ( bind Rust.Mutable vec (chain "into_vec" [] (Rust.Var str))
                        : exprToStatements (chain "push" [
                                Rust.Lit (Rust.LitByteChar '\NUL')
                            ] (Rust.Var vec))
                        ) (Just (Rust.Var vec))))
                ] $
                call "::std::env::args_os" []
```

Cでは、 `argv`は変更可能な文字列の変更可能なNULL終端配列である必要があります。
`argv_storage`には配列を基にしたベクトルとして修正可能な文字列があるので、今度はそれらの文字列へのポインタを配列に基いたベクトルを構築する必要があります。

これらのポインタを保存したら、元の配列を再割り当てしてポインタを無効にする可能性があるため、取得元のベクトルの長さを変更する可能性があるものは何もしないでください。

[`Iterator.chain`]（https://doc.rust-lang.org/stable/std/iter/trait.Iterator.html#method.chain）は元のイテレータのすべての要素を生成する新しいイテレータを生成します。 2番目のイテラブル内の何かが続きます。
配列の最後にNULLポインタを1つ追加したいだけで、便利なことに、Option型は反復可能です。

```haskell
            , bind Rust.Mutable argv $
                chain "collect::<Vec<_>>" [] $
                chain "chain" [call "Some" [call "::std::ptr::null_mut" []]] $
                chain "map" [
                    Rust.Lambda [vec] (chain "as_mut_ptr" [] (Rust.Var vec))
                ] $
                chain "iter_mut" [] $
                Rust.Var argv_storage
            ]
```

Cでは、 `argc`は` argv`配列の要素数ですが、終端のNULLポインタは数えません。
それで、代わりに `argv_storage`にある項目の数を渡します。

`main`の2番目の引数には、` argv`を裏付ける配列へのポインタを渡します。
その後、 `argv`の長さを変更する可能性のあることは何もしないでください、そして本当の` main`が戻るまで `argv`の割り当てが解除されないようにしなければなりません。

```haskell
        args =
            [ Rust.Cast (chain "len" [] (Rust.Var argv_storage)) (toRustType argcType)
            , chain "as_mut_ptr" [] (Rust.Var argv)
            ]
```

期待するすべての引数を構成する方法がわからない限り、main関数を含むプログラムを翻訳することはできません。

```haskell
    wrapArgv _ = unimplemented declr
```

`argc`と` argv`を処理した後、慣習的に `envp`と呼ばれる3番目の引数を見るかもしれません。
POSIXシステムでは、この引数に `environ`という名前のグローバル変数に格納されたポインタを渡すことができます。

```haskell
    wrapEnvp [] = return ([], [])
    wrapEnvp [arg@(IsPtr Rust.Mutable (IsPtr Rust.Mutable ty))] | ty == charType
        = return (setup, args)
        where
        environ = Rust.VarName "environ"
        setup =
            [ Rust.StmtItem [] $
                Rust.Extern [Rust.ExternStatic Rust.Immutable environ (toRustType arg)]
            ]
        args = [Rust.Var environ]
    wrapEnvp _ = unimplemented declr
```


ステートメント
==========

`break`と` continue`ステートメントを解釈するためには、もう少しコンテキストを追加する必要があります。それについては後で説明します。
そのコンテキストを単純なデータ型でラップしましょう。

```haskell
data OuterLabels = OuterLabels
    { onBreak :: Maybe Label
    , onContinue :: Maybe Label
    , switchExpression :: Maybe CExpr
    }

newtype SwitchCases = SwitchCases (IntMap.IntMap (Maybe Result))

instance Monoid SwitchCases where
    mempty = SwitchCases IntMap.empty
    SwitchCases a `mappend` SwitchCases b = SwitchCases $
        IntMap.unionWith (liftM2 eitherCase) a b
        where
        eitherCase lhs rhs = Result
            { resultType = IsBool
            , resultMutable = Rust.Immutable
            , result = Rust.LOr (toBool lhs) (toBool rhs)
            }
```

関数の中にはCの文があります。
ステートメントを中心としたCの構文とは異なり、Rustの構文はほとんど単なる式です。
それにもかかわらず、（単にRust式ではなく） `interpretStatement`にRustステートメントのリストを返させることで、余分な波括弧を取り除くことができます。

```haskell
type CSourceBuildCFGT s = BuildCFGT (RWST OuterLabels SwitchCases (Map.Map Ident Label) (EnvMonad s)) [Rust.Stmt] Result

interpretStatement :: CStat -> CSourceBuildCFGT s ([Rust.Stmt], Terminator Result) -> CSourceBuildCFGT s ([Rust.Stmt], Terminator Result)
```

```haskell
interpretStatement (CLabel ident body _ _) next = do
    label <- gotoLabel ident
    (rest, end) <- interpretStatement body next
    addBlock label rest end
    return ([], Branch label)
```

```haskell
interpretStatement stmt@(CCase expr body node) next = do
    selector <- getSwitchExpression stmt
    let condition = CBinary CEqOp selector expr node
    addSwitchCase (Just condition) body next
interpretStatement stmt@(CCases lower upper body node) next = do
    selector <- getSwitchExpression stmt
    let condition = CBinary CLndOp
            (CBinary CGeqOp selector lower node)
            (CBinary CLeqOp selector upper node)
            node
    addSwitchCase (Just condition) body next
interpretStatement (CDefault body _) next =
    addSwitchCase Nothing body next
```

Cステートメントは、Cステートメントの後にセミコロンが続く「ステートメント式」のように単純なものです。
その場合、式を翻訳するだけで文を翻訳できます。

単なるセミコロンのように文が空の場合は、文を生成する必要はありません。

```haskell
interpretStatement (CExpr Nothing _) next = next
```

そうでなければ、 `interpretExpr`の最初の引数は式が結果が問題となるコンテキストで現れるかどうかを示します。
ステートメント式では、結果は破棄されるので、 `False`を渡します。

```haskell
interpretStatement (CExpr (Just expr) _) next = do
    expr' <- lift $ lift $ interpretExpr False expr
    (rest, end) <- next
    return (resultToStatements expr' ++ rest, end)
```

「ブロック」とも呼ばれる「複合文」を使用できます。
複合ステートメントは、ゼロ個以上のステートメントまたは宣言のシーケンスを含みます。詳細は後述の `interpretBlockItem`を参照してください。

空の複合ステートメントの場合は、生成されたRustを単純化しないという通常の規則を例外とし、単に「ステートメント」を無視します。

```haskell
interpretStatement (CCompound [] items _) next = mapBuildCFGT (mapRWST scope) $ do
    foldr interpretBlockItem next items
```

このステートメントは、条件式と "then"ブランチを持つ `if`ステートメント、そしておそらく" else "ブランチです。

条件式はCでは数値ですが、Rustでは `bool`型でなければなりません。
必要ならば、式を変換するために以下に定義されている `toBool`ヘルパーを使います。

Cでは、 "then"と "else"の分岐はそれぞれステートメント（中括弧で囲まれている場合は複合ステートメントになる可能性があります）ですが、Rustではそれぞれ分岐である必要があります（したがってそれらは常に中括弧で囲まれています） ）

ここでは両方のブランチをブロックに強制するために `statementsToBlock`を使っていますので、元のプログラムがそのブランチで複合文を使っているかどうかは気にしません。

```haskell
interpretStatement (CIf c t mf _) next = do
    c' <- lift $ lift $ interpretExpr True c
    after <- newLabel

    falseLabel <- case mf of
        Nothing -> return after
        Just f -> do
            (falseEntry, falseTerm) <- interpretStatement f (return ([], Branch after))
            falseLabel <- newLabel
            addBlock falseLabel falseEntry falseTerm
            return falseLabel

    (trueEntry, trueTerm) <- interpretStatement t (return ([], Branch after))
    trueLabel <- newLabel
    addBlock trueLabel trueEntry trueTerm

    (rest, end) <- next
    addBlock after rest end

    return ([], CondBranch c' trueLabel falseLabel)
```

```haskell
interpretStatement stmt@(CSwitch expr body node) next = do
    (bindings, expr') <- case expr of
        CVar{} -> return ([], expr)
        _ -> lift $ lift $ do
            ident <- fmap internalIdent (uniqueName "switch")
            rhs <- interpretExpr True expr
            var <- addSymbolIdent ident (Rust.Immutable, resultType rhs)
            return
                ( [Rust.Let Rust.Immutable (Rust.VarName var) Nothing (Just (result rhs))]
                , CVar ident node
                )

    after <- newLabel
    (_, SwitchCases cases) <- getSwitchCases expr' $ setBreak after $
        interpretStatement body (return ([], Branch after))

    let isDefault (Just condition) = Left condition
        isDefault Nothing = Right ()
    let (conditions, defaults) = IntMap.mapEither isDefault cases
    defaultCase <- case IntMap.keys defaults of
        [] -> return after
        [defaultCase] -> return defaultCase
        _ -> lift $ lift $ badSource stmt "duplicate default cases"

    entry <- foldrM conditionBlock defaultCase (IntMap.toList conditions)

    (rest, end) <- next
    addBlock after rest end

    return (bindings, Branch entry)
    where
    conditionBlock (target, condition) defaultCase = do
        label <- newLabel
        addBlock label [] (CondBranch condition target defaultCase)
        return label
```

whileループは、CからRustへの変換が簡単です。
ループ条件の型が `bool`で、ループ本体がブロックでなければならない` if`文に類似した違いを除いて、それらは両方の言語で同じ意味を持ちます。

```haskell
interpretStatement (CWhile c body doWhile _) next = do
    c' <- lift $ lift $ interpretExpr True c
    after <- newLabel

    headerLabel <- newLabel
    (bodyEntry, bodyTerm) <- setBreak after $ setContinue headerLabel $
        interpretStatement body (return ([], Branch headerLabel))

    bodyLabel <- newLabel
    addBlock bodyLabel bodyEntry bodyTerm

    addBlock headerLabel [] $ case toBool c' of
        Rust.Lit (Rust.LitBool cont) | cont /= doWhile ->
            Branch (if cont then bodyLabel else after)
        _ -> CondBranch c' bodyLabel after

    (rest, end) <- next
    addBlock after rest end

    return ([], Branch (if doWhile then bodyLabel else headerLabel))
```

Cの `for`ループはRustに変換するのが難しいことがあります。

ああ、ループの初期化/宣言部分は十分に簡単です。
新しいスコープを開き、その先頭に代入と `let`-bindingsを挿入すれば、うまくいきます。
`let`バインディングがある場合はすべてをブロックでラップする必要がありますが、それ以外は必要ありません。

```haskell
interpretStatement (CFor initial mcond mincr body _) next = do
    after <- newLabel

    ret <- mapBuildCFGT (mapRWST scope) $ do
        prefix <- case initial of
            Left Nothing -> return []
            Left (Just expr) -> do
                expr' <- lift $ lift $ interpretExpr False expr
                return (resultToStatements expr')
            Right decls -> lift $ lift $ interpretDeclarations makeLetBinding decls

        headerLabel <- newLabel
        incrLabel <- case mincr of
            Nothing -> return headerLabel
            Just incr -> do
                incr' <- lift $ lift $ interpretExpr False incr
                incrLabel <- newLabel
                addBlock incrLabel (resultToStatements incr') (Branch headerLabel)
                return incrLabel

        (bodyEntry, bodyTerm) <- setBreak after $ setContinue incrLabel $
            interpretStatement body (return ([], Branch incrLabel))

        bodyLabel <- newLabel
        addBlock bodyLabel bodyEntry bodyTerm

        cond <- case mcond of
            Just cond -> do
                cond' <- lift $ lift $ interpretExpr True cond
                return (CondBranch cond' bodyLabel after)
            Nothing -> return (Branch bodyLabel)
        addBlock headerLabel [] cond

        return (prefix, Branch headerLabel)

    (rest, end) <- next
    addBlock after rest end

    return ret
```

```haskell
interpretStatement (CGoto ident _) next = do
    _ <- next
    label <- gotoLabel ident
    return ([], Branch label)
```

`continue`と` break`ステートメントは周囲の文脈で決定したどんな表現にも変換されます。
例えば、一番近い囲んでいるループがforループであれば、 `continue`は` break 'continueTo`に変換されます。

```haskell
interpretStatement stmt@(CCont _) next = do
    _ <- next
    val <- lift (asks onContinue)
    case val of
        Just label -> return ([], Branch label)
        Nothing -> lift $ lift $ badSource stmt "continue outside loop"
interpretStatement stmt@(CBreak _) next = do
    _ <- next
    val <- lift (asks onBreak)
    case val of
        Just label -> return ([], Branch label)
        Nothing -> lift $ lift $ badSource stmt "break outside loop"
```

`return`ステートメントは、とても単純明快な方法です。もしあれば返すべき式を翻訳し、Rustと同等のものを返します -  RustはCよりも型についてより厳密です。
そのため、return式が関数の宣言されたreturn型とは異なる型を持つ場合は、正しい型への型キャストを挿入する必要があります。

```haskell
interpretStatement stmt@(CReturn expr _) next = do
    _ <- next
    lift $ lift $ do
        val <- lift (asks functionReturnType)
        case val of
            Nothing -> badSource stmt "return statement outside function"
            Just retTy -> do
                expr' <- mapM (fmap (castTo retTy) . interpretExpr True) expr
                return (exprToStatements (Rust.Return expr'), Unreachable)
```

それ以外の場合、これはまだ翻訳を実装していない種類のステートメントです。

> ** TODO **：もっと多くの種類の文を翻訳する。 `:-)`

```haskell
interpretStatement stmt _ = lift $ lift $ unimplemented stmt
```

上記のループの内側で、 `break`または` continue`ステートメントが現れた場合に使うために翻訳を更新する必要がありました。
これらの関数は更新された `break` /` continue`式を使って提供された変換アクションを実行します。

```haskell
setBreak :: Label -> CSourceBuildCFGT s a -> CSourceBuildCFGT s a
setBreak label =
    mapBuildCFGT (local (\ flow -> flow { onBreak = Just label }))

setContinue :: Label -> CSourceBuildCFGT s a -> CSourceBuildCFGT s a
setContinue label =
    mapBuildCFGT (local (\ flow -> flow { onContinue = Just label }))
```

```haskell
getSwitchExpression :: CStat -> CSourceBuildCFGT s CExpr
getSwitchExpression stmt = do
    mexpr <- lift $ asks switchExpression
    case mexpr of
        Nothing -> lift $ lift $ badSource stmt "case outside switch"
        Just expr -> return expr

addSwitchCase :: Maybe CExpr -> CStat -> CSourceBuildCFGT s ([Rust.Stmt], Terminator Result) -> CSourceBuildCFGT s ([Rust.Stmt], Terminator Result)
addSwitchCase condition body next = do
    condition' <- lift $ lift $ mapM (interpretExpr True) condition
    next' <- interpretStatement body next
    label <- case next' of
        ([], Branch to) -> return to
        (rest, end) -> do
            label <- newLabel
            addBlock label rest end
            return label
    lift $ tell $ SwitchCases $ IntMap.singleton label condition'
    return ([], Branch label)

getSwitchCases :: CExpr -> CSourceBuildCFGT s a -> CSourceBuildCFGT s (a, SwitchCases)
getSwitchCases expr = mapBuildCFGT wrap
    where
    wrap body = do
        ((a, st), cases) <- censor (const mempty)
            $ local (\ flow -> flow { switchExpression = Just expr })
            $ listen body
        return ((a, cases), st)
```

Cでは `goto`文とそれに対応するラベルを任意の順序で並べることができます。
そのため、Cラベルの名前に最初に出会ったときは、ラベル付きの文であるか `goto`の文であるかにかかわらず、CFGラベルを割り当てます。
次に、選択したCFGラベルを保存して、将来の参照に必ず同じラベルを使用できるようにします。

```haskell
gotoLabel :: Ident -> CSourceBuildCFGT s Label
gotoLabel ident = do
    labels <- lift get
    case Map.lookup ident labels of
        Nothing -> do
            label <- newLabel
            lift (put (Map.insert ident label labels))
            return label
        Just label -> return label
```

一連のステートメントの制御フローグラフを作成したら、同等のRust制御フロー式を抽出する必要があります。

```haskell
cfgToRust :: (Pretty node, Pos node) => node -> CSourceBuildCFGT s ([Rust.Stmt], Terminator Result) -> EnvMonad s [Rust.Stmt]
cfgToRust _node build = do
    let builder = buildCFG $ do
            (early, term) <- build
            entry <- newLabel
            addBlock entry early term
            return entry
    (rawCFG, _) <- evalRWST builder (OuterLabels Nothing Nothing Nothing) Map.empty
```

元のプログラムにはない新しい変数を導入したり、コードの一部を複製しなければ、これは常に可能とは限りません。
現時点では、そのような場合には翻訳エラーを報告することを選択します。他の制御フローパターンについては、まだ認識方法がわかりません。

```haskell
    let cfg = depthFirstOrder (removeEmptyBlocks rawCFG)
    let (hasGoto, structured) = structureCFG mkBreak mkContinue mkLoop mkIf mkGoto mkMatch cfg
    return $ if hasGoto then declCurrent : structured else structured
    where
```

`CFG`モジュールはソース言語とターゲット言語の両方にとらわれないので、どの共通パターンがRustに適用されるのか、そしてどのように適切なRust ASTを構築するのかをそれに伝える必要があります。

このリストの順番は重要です。
特定のパターンに一致するすべてのブロックは、次のパターンを試す前に変換されます。
さもなければ完璧な `while`ループが` loop`の中にネストされた `if`に変換されるかもしれないので、` ifThenElse`は `while`の後に来ることが重要です。

```haskell
    loopLabel l = Rust.Lifetime ("loop" ++ show l)
    mkBreak l = exprToStatements (Rust.Break (fmap loopLabel l))
    mkContinue l = exprToStatements (Rust.Continue (fmap loopLabel l))
    mkLoop l b = exprToStatements (Rust.Loop (Just (loopLabel l)) (statementsToBlock b))
    mkIf c t f = exprToStatements (simplifyIf c (statementsToBlock t) (statementsToBlock f))

    currentBlock = Rust.VarName "_currentBlock"
    declCurrent = Rust.Let Rust.Mutable currentBlock Nothing Nothing
    mkGoto l = exprToStatements (Rust.Assign (Rust.Var currentBlock) (Rust.:=) (fromIntegral l))
    mkMatch = flip (foldr go)
        where
        go (l, t) f = exprToStatements (Rust.IfThenElse (Rust.CmpEQ (Rust.Var currentBlock) (fromIntegral l)) (statementsToBlock t) (statementsToBlock f))
```

できるだけ明確なコードを生成するために、 `if`ステートメントのためのいくつかの興味深い特別なケースを扱います。

-  trueとfalseの両方のブランチが空の場合、 `if`ステートメントはまったく必要ありません。
私たちはただその副作用について状態を評価する必要があります。

    ```haskell
        simplifyIf c (Rust.Block [] Nothing) (Rust.Block [] Nothing) =
            result c
    ```

- 真の分岐だけが空の場合は、条件の逆を計算して、分岐を交換する必要があります。
そうでなければ、 `if ... {} else {...}`のような表現になるでしょう。
このパターンは、いくつかの `continue`ステートメントの翻訳など、驚くべき場所に現れることがあります。

    ```haskell
        simplifyIf c (Rust.Block [] Nothing) f =
            Rust.IfThenElse (toNotBool c) f (Rust.Block [] Nothing)
    ```

- そうでなければ、通常の `if`ステートメントを構築するだけです。

    ```haskell
        simplifyIf c t f = Rust.IfThenElse (toBool c) t f
    ```


ブロックとスコープ
=================

複合文（上記を参照）では、入れ子になった文だけでなくローカル宣言もあります。
interpretBlockItemは、各複合ブロック項目に対して0個以上のRustステートメントのシーケンスを生成します。

```haskell
interpretBlockItem :: CBlockItem -> CSourceBuildCFGT s ([Rust.Stmt], Terminator Result) -> CSourceBuildCFGT s ([Rust.Stmt], Terminator Result)
interpretBlockItem (CBlockStmt stmt) next = interpretStatement stmt next
interpretBlockItem (CBlockDecl decl) next = do
    decl' <- lift $ lift (interpretDeclarations makeLetBinding decl)
    (rest, end) <- next
    return (decl' ++ rest, end)
interpretBlockItem item _ = lift $ lift (unimplemented item)
```

`scope`は` m`の変換ステップを実行しますが、その後 `m`が環境に加えた変更をすべて捨てます。
ただし、グローバル状態の変更と同様に、出力に追加された項目は保持されます。

```haskell
scope :: EnvMonad s a -> EnvMonad s a
scope m = do
    -- Save the current environment.
    old <- lift get
    a <- m
    -- Restore the environment to its state before running m.
    lift (modify (\ st -> old { globalState = globalState st }))
    return a
```


ブロックと文のためのスマートコンストラクタ
--------------------------------------------

時にはブロックを生成して、そのブロックの最終結果を使用しないことに気付くことがあります。
その場合、ブロックの最後に最後の式があると、それをステートメントに変換し、それを残りのステートメントに追加することができます。

```haskell
blockToStatements :: Rust.Block -> [Rust.Stmt]
blockToStatements (Rust.Block stmts mexpr) = case mexpr of
    Just expr -> stmts ++ exprToStatements expr
    Nothing -> stmts
```

リスト内の唯一のステートメントがそれ自体でブロックである場合、ステートメントのリストを囲むように新しいブロックをラップする意味はありません。

ブロックの最終式がない場合は、 `Rust.Block`コンストラクタの代わりにこの関数を使ってください。

```haskell
statementsToBlock :: [Rust.Stmt] -> Rust.Block
statementsToBlock [Rust.Stmt (Rust.BlockExpr stmts)] = stmts
statementsToBlock stmts = Rust.Block stmts Nothing
```

結果が使われないときにどうやって `if`式を生成するかについて特に注意しなければなりません。
ほとんどの式がステートメントとして使用されている場合、結果が無視されるため、式の型は関係ありません。
しかし、Rustのif式がステートメントとして使用されている場合、Rustはその型が（）であることを要求します。
trueとfalseの分岐をブロック最終式として配置するのではなく、ステートメントにラップすることで型が正しいことを確認できます。

また、2つのスタック割り当て変数が同じ名前を持たないと仮定すると、ステートメントとしてブロック式を使用する必要はありません。

必要な不変式がどこでも維持されるようにするために `Rust.Stmt`コンストラクタを直接使う代わりにこの関数を使ってください。

```haskell
exprToStatements :: Rust.Expr -> [Rust.Stmt]
exprToStatements (Rust.IfThenElse c t f) =
    [Rust.Stmt (Rust.IfThenElse c (extractExpr t) (extractExpr f))]
    where
    extractExpr = statementsToBlock . blockToStatements
exprToStatements (Rust.BlockExpr b) = blockToStatements b
exprToStatements e = [Rust.Stmt e]
```


式
===========

式の変換は、式ツリーをボトムアップでたどることによって機能します。サブ式の型を計算し、それを使用して周囲の式の意味を判断します。

また、式が正当な "l-value"であるかどうかを記録する必要があります。これは、その式に代入できるか、それとも可変の借用が可能かを決定します。
ここでは `Rust.Mutable`型を悪用しています。それが` Mutable`なら、それは正当なL値であり、 `Immutable`ならばそうではありません。

最後に、もちろん、副式のRust ASTを記録して、構築する大きな式に結合できるようにします。

```haskell
data Result = Result
    { resultType :: CType
    , resultMutable :: Rust.Mutable
    , result :: Rust.Expr
    }
```

`resultToStatements`は` exprToStatements`を持ち上げて `Result`を操作するための便利なラッパーです。

```haskell
resultToStatements :: Result -> [Rust.Stmt]
resultToStatements = exprToStatements . result
```

```haskell
typeToResult :: IntermediateType -> Rust.Expr -> Result
typeToResult itype expr = Result
    { resultType = typeRep itype
    , resultMutable = typeMutable itype
    , result = expr
    }
```

`interpretExpr`は` Result`の他のメタデータと共に `CExpression`をRust式に変換します。

呼び出し側がこの式の結果を使用するつもりであるかどうかを示すブール値のパラメータ `demand`も取ります。
一部のC式では、結果が必要な場合は追加のRustコードを生成する必要があります。

```haskell
interpretExpr :: Bool -> CExpr -> EnvMonad s Result
```

Cのコンマ演算子は、左側の式の副作用を評価し、結果を破棄してから、右側の式の結果に評価します。

Rustには専用の演算子がありませんが、Rustのブロック式と同じように機能します。 `（e1、e2、e3）`は `{e1;}に変換できます。 e2; e3}。
結果が使用されないのであれば、代わりに `{e1; e2; e3;ここで、式はセミコロンで区切られるのではなく、セミコロンで終了します。

```haskell
interpretExpr demand (CComma exprs _) = do
    let (effects, mfinal) = if demand then (init exprs, Just (last exprs)) else (exprs, Nothing)
    effects' <- mapM (fmap resultToStatements . interpretExpr False) effects
    mfinal' <- mapM (interpretExpr True) mfinal
    return Result
        { resultType = maybe IsVoid resultType mfinal'
        , resultMutable = maybe Rust.Immutable resultMutable mfinal'
        , result = Rust.BlockExpr (Rust.Block (concat effects') (fmap result mfinal'))
        }
```

Cの代入演算子は非常に複雑で、左辺と右辺を再帰的に変換した後、以下で定義されている `compound`ヘルパー関数に委譲するだけです。

```haskell
interpretExpr demand expr@(CAssign op lhs rhs _) = do
    lhs' <- interpretExpr True lhs
    rhs' <- interpretExpr True rhs
    compound expr False demand op lhs' rhs'
```

Cの三項条件演算子（ `c？t：f`）は、Rustの` if` / `else`式にかなり直接変換されます。

```haskell
interpretExpr demand expr@(CCond c (Just t) f _) = do
    c' <- fmap toBool (interpretExpr True c)
    t' <- interpretExpr demand t
    f' <- interpretExpr demand f
    if demand
        then promotePtr expr (mkIf c') t' f'
        else return Result
            { resultType = IsVoid
            , resultMutable = Rust.Immutable
            , result = mkIf c' (result t') (result f')
            }
    where
    mkIf c' t' f' = Rust.IfThenElse c' (Rust.Block [] (Just t')) (Rust.Block [] (Just f'))
```

Cの二項演算子は非常に複雑であるため、左辺と右辺を再帰的に変換した後、以下に定義する `binop`ヘルパー関数に委譲するだけです。

```haskell
interpretExpr _ expr@(CBinary op lhs rhs _) = do
    lhs' <- interpretExpr True lhs
    rhs' <- interpretExpr True rhs
    binop expr op lhs' rhs'
```

Cのキャスト演算子は、Rustのキャスト演算子とまったく同じです。
構文はかなり異なりますが、意味は同じです。
キャストの結果が正しいl値になることはありません。

Cでは、式の結果を `void`にキャストすることは結果を明示的に無視するためのイディオムです。つまり、式はその副作用についてのみ評価されることを意図しています。
古いコードでは、過度に熱心なコンパイラ警告を抑制するために一般的に使用されており、glibcのassertマクロの実装など、特定の特殊なケースでは引き続き使用されています。
この慣用句の単純な翻訳では無効なRustが生成されるため、その結果を要求せずに単純に部分式を評価することを特別なケースとして扱います。

```haskell
interpretExpr _ (CCast decl expr _) = do
    (_mut, ty) <- typeName decl
    expr' <- interpretExpr (ty /= IsVoid) expr
    return Result
        { resultType = ty
        , resultMutable = Rust.Immutable
        , result = (if ty == IsVoid then result else castTo ty) expr'
        }
```

プレ/ポストインクリメント/デクリメント演算子を複合代入演算子に非糖化し、共通の代入ヘルパーを呼び出します。

```haskell
interpretExpr demand node@(CUnary op expr _) = case op of
    CPreIncOp -> incdec False CAddAssOp
    CPreDecOp -> incdec False CSubAssOp
    CPostIncOp -> incdec True CAddAssOp
    CPostDecOp -> incdec True CSubAssOp
```

Cのaddress-of演算子（単項接頭辞 `＆`）を可変または不変の借り演算子に変換し、その後に対応するRust生ポインタ型にキャストします。

これがミュータブルかイミュータブルかを判断するためにl-valueフラグを再利用します。
部分式が有効なL値である場合は、可変のポインタを使用しても安全です。
そうでなければ、不変ポインタを使っても大丈夫かもしれませんし、無意味かもしれません。

この実装はナンセンスなケースをチェックしません。
変な結果が得られる場合は、実際のCコンパイラを使用して、入力Cがエラーなしでコンパイルされることを確認してください。

```haskell
    CAdrOp -> do
        expr' <- interpretExpr True expr
        let ty' = IsPtr (resultMutable expr') (resultType expr')
        return Result
            { resultType = ty'
            , resultMutable = Rust.Immutable
            , result = Rust.Cast (Rust.Borrow (resultMutable expr') (result expr')) (toRustType ty')
            }
```

Cの間接演算子または間接参照演算子（単項接頭辞 `*`）は、Rustでは同じ演算子に変換されます。
結果がl値であるかどうかは、ポインタが可変値を指していたかどうかによって異なります。

```haskell
    CIndOp -> do
        expr' <- interpretExpr True expr
        case resultType expr' of
            IsPtr mut' ty' -> return Result
                { resultType = ty'
                , resultMutable = mut'
                , result = Rust.Deref (result expr')
                }
            IsFunc{} -> return expr'
            _ -> badSource node "dereference of non-pointer"
```

ああ、単項プラス。
お気に入り。
この演算子はほとんど役に立ちませんので、Rustには単項プラス演算子がありません。
これは単項マイナスの算術演算子です。その演算子がその引数を否定する場合、この演算子はその引数を変更せずに返します。 ...「整数の昇格」規則が適用されることを除いて、この演算子は暗黙のキャストを実行するかもしれません。

```haskell
    CPlusOp -> do
        expr' <- interpretExpr demand expr
        let ty' = intPromote (resultType expr')
        return Result
            { resultType = ty'
            , resultMutable = Rust.Immutable
            , result = castTo ty' expr'
            }
```

Cの単項マイナス演算子は、整数の昇格を適用した後でRustの単項マイナス演算子に変換されます。符号なし型を否定する場合、Cはwrapに整数オーバーフローを定義します。

```haskell
    CMinOp -> fmap wrapping $ simple Rust.Neg
```

ビットごとの補数はRustの `！`演算子に変換されます。

```haskell
    CCompOp -> simple Rust.Not
```

論理的な "not"はRustの `！`演算子にも変換されますが、最初にオペランドを `bool`型にする必要があります。
反対の値を返す `toBool`の特別な場合の` toNotBool`バリアントを作成することで愚かな余分な "not"演算子を導入することを避けることができます。

```haskell
    CNegOp -> do
        expr' <- interpretExpr True expr
        return Result
            { resultType = IsBool
            , resultMutable = Rust.Immutable
            , result = toNotBool expr'
            }
```

単項演算子用の一般的なヘルパー：

```haskell
    where
    incdec returnOld assignop = do
        expr' <- interpretExpr True expr
        compound node returnOld demand assignop expr' Result
            { resultType = IsInt Signed (BitWidth 32)
            , resultMutable = Rust.Immutable
            , result = 1
            }
    simple f = do
        expr' <- interpretExpr True expr
        let ty' = intPromote (resultType expr')
        return Result
            { resultType = ty'
            , resultMutable = Rust.Immutable
            , result = f (castTo ty' expr')
            }
```

Cの `sizeof`と` alignof`演算子は、それぞれ `std :: mem`の中のRustの` size_of`と `align_of`関数の呼び出しに変換されます。
式については、評価するのではなく、その型を決定するだけです。

> ** TODO **：Rustの `std :: mem :: size_of_val`を使用して、可変長配列を` sizeof`（起こりうる副作用も含む）について評価する必要があります。
>可変長配列に対してalignofは何をすべきですか？

```haskell
interpretExpr _ (CSizeofExpr e _) = do
    e' <- interpretExpr True e
    return (rustSizeOfType (toRustType (resultType e')))
interpretExpr _ (CSizeofType decl _) = do
    (_mut, ty) <- typeName decl
    return (rustSizeOfType (toRustType ty))
interpretExpr _ (CAlignofExpr e _) = do
    e' <- interpretExpr True e
    return (rustAlignOfType (toRustType (resultType e')))
interpretExpr _ (CAlignofType decl _) = do
    (_mut, ty) <- typeName decl
    return (rustAlignOfType (toRustType ty))
```

Cの配列インデックスまたは配列添字演算子 `e1 [e2]`は、2つの式を追加してその結果を間接参照するのと同じように機能します。
C式の `e1 [e2]`では、通常 `e1`がポインタで` e2`が整数であると期待されていますが、Cはそれらを逆にすることを許可します。
幸いなことに、Cのポインタ加算も交換可能なので、ここで `binop`ヘルパーを呼び出すことは正しいことをします。

```haskell
interpretExpr _ expr@(CIndex lhs rhs _) = do
    lhs' <- interpretExpr True lhs
    rhs' <- interpretExpr True rhs
    case (resultType lhs', resultType rhs') of
        (IsArray mut _ el, _) -> return (subscript mut el (result lhs') rhs')
        (_, IsArray mut _ el) -> return (subscript mut el (result rhs') lhs')
        _ -> do
            ptr <- binop expr CAddOp lhs' rhs'
            case resultType ptr of
                IsPtr mut ty -> return Result
                    { resultType = ty
                    , resultMutable = mut
                    , result = Rust.Deref (result ptr)
                    }
                _ -> badSource expr "array subscript of non-pointer"
    where
    subscript mut el arr idx = Result
        { resultType = el
        , resultMutable = mut
        , result = Rust.Index arr (castTo (IsInt Unsigned WordWidth) idx)
        }
```

関数呼び出しは、最初にどの関数を呼び出すかを識別する式、および引数式を変換します。

```haskell
interpretExpr _ expr@(CCall func args _) = do
    func' <- interpretExpr True func
    case resultType func' of
        IsFunc retTy argTys variadic -> do
            args' <- castArgs variadic (map snd argTys) args
            return Result
                { resultType = retTy
                , resultMutable = Rust.Immutable
                , result = Rust.Call (result func') args'
                }
        _ -> badSource expr "function call to non-function"
    where
```

実パラメータごとに、そのパラメータに指定された式が変換されてから（必要に応じて）対応する仮パラメータの型にキャストされます。

可変関数を呼び出す場合、関数の型で明示的に指定された引数の後に引数をいくつでも配置できます。それらの追加の引数は任意の型にすることができます。

それ以外の場合は、関数の型と同じ数の引数が必要です。そうしないと、構文エラーになります。

```haskell
    castArgs _ [] [] = return []
    castArgs variadic (ty : tys) (arg : rest) = do
        arg' <- interpretExpr True arg
        args' <- castArgs variadic tys rest
        return (castTo ty arg' : args')
    castArgs True [] rest = mapM (fmap promoteArg . interpretExpr True) rest
    castArgs False [] _ = badSource expr "arguments (too many)"
    castArgs _ _ [] = badSource expr "arguments (too few)"
```

Cでは、「デフォルト引数の昇格」（C99 6.5.2.2段落6〜7）は、最後に宣言されたパラメータの後にあるすべての変数パラメータに適用されます。
空の引数リスト（ `foo（）`）で宣言された関数や、プロトタイプがないために暗黙的に宣言された関数に渡された引数にも適用されます。

```haskell
    promoteArg :: Result -> Rust.Expr
    promoteArg r = case resultType r of
        IsFloat _ -> castTo (IsFloat 64) r
        IsArray mut _ el -> castTo (IsPtr mut el) r
        ty -> castTo (intPromote ty) r
```

構造体メンバへのアクセスはC言語では2つの形式（ `.`と`  - > `）を持ちますが、それは左側が先に間接参照されるかどうかという点でのみ異なります。
`p-> f`を`（* p）.f`に変換し、それから `o.f`をRustの同じものに変換します。
結果の型は、構造体内の名前付きフィールドの型です。結果は、大きい方のオブジェクトが正当なL値である場合に限り、正当なL値になります。

```haskell
interpretExpr _ expr@(CMember obj ident deref node) = do
    obj' <- interpretExpr True $ if deref then CUnary CIndOp obj node else obj
    objTy <- completeType (resultType obj')
    fields <- case objTy of
        IsStruct _ fields -> return fields
        _ -> badSource expr "member access of non-struct"
    let field = applyRenames ident
    ty <- case lookup field fields of
        Just ty -> return ty
        Nothing -> badSource expr "request for non-existent field"
    return Result
        { resultType = ty
        , resultMutable = resultMutable obj'
        , result = Rust.Member (result obj') (Rust.VarName field)
        }
```

変換中は変数名を保持するので、C変数参照は同一のRust変数参照に変換されます。
しかし、その型とそのアドレスを取ることで可変ポインタまたは不変ポインタを生成するかどうかを報告するために、環境内で変数を検索する必要があります。

```haskell
interpretExpr _ expr@(CVar ident _) = do
    sym <- getSymbolIdent ident
    maybe (badSource expr "undefined variable") return sym
```

Cリテラル（整数、浮動小数点、文字、および文字列）は、Rustでは同様のトークンに変換されます。

> ** TODO **：Rustに言えばできる限り浮動小数点の16進数リテラルについてどうするかを考え出してください（まだ？）。

> ** TODO **：ワイド文字と文字列リテラルを翻訳します。

```haskell
interpretExpr _ expr@(CConst c) = case c of
```

Cでは、整数リテラルの型は、その値がどの型に収まるかによって異なり、接尾辞（ `U`または` L`）と、その表現が10進数かそれ以外の基数かによって制約されます。
C99 6.4.4.1段落5およびそれに続く表を参照。

リテラルが型の範囲内に収まるかどうかを判断するために、 `long`は32ビットであると見なすことにしましたが、それを与えるRust型は` isize`です。
定数が32ビットに収まらない場合は、常に型 `i64`を指定します。

```haskell
    CIntConst (CInteger v repr flags) _ ->
        let allow_signed = not (testFlag FlagUnsigned flags)
            allow_unsigned = not allow_signed || repr /= DecRepr
            widths =
                [ (32 :: Int,
                    if any (`testFlag` flags) [FlagLongLong, FlagLong]
                    then WordWidth else BitWidth 32)
                , (64, BitWidth 64)
                ]
            allowed_types =
                [ IsInt s w
                | (bits, w) <- widths
                , (True, s) <- [(allow_signed, Signed), (allow_unsigned, Unsigned)]
                , v < 2 ^ (bits - if s == Signed then 1 else 0)
                ]
            repr' = case repr of
                DecRepr -> Rust.DecRepr
                OctalRepr -> Rust.OctalRepr
                HexRepr -> Rust.HexRepr
        in case allowed_types of
        [] -> badSource expr "integer (too big)"
        ty : _ -> return (literalNumber ty (Rust.LitInt v repr'))
    CFloatConst (CFloat str) _ -> case span (`notElem` "fF") str of
        (v, "") -> return (literalNumber (IsFloat 64) (Rust.LitFloat v))
        (v, [_]) -> return (literalNumber (IsFloat 32) (Rust.LitFloat v))
        _ -> badSource expr "float"
```

Rustの `char`型はUnicode文字用なので、Cの8ビットの` char`型とはかなり異なります。
結果として、Cの文字リテラルと文字列は、より一般的な文字と文字列リテラルの構文ではなく、Rustの "バイトリテラル"（ `b '？'`、 `b" ... "`）に変換する必要があります。

```haskell
    CCharConst (CChar ch False) _ -> return Result
        { resultType = charType
        , resultMutable = Rust.Immutable
        , result = Rust.Lit (Rust.LitByteChar ch)
        }
```

Cでは、文字列リテラルは末尾に追加される末尾のNUL文字を取得します。
さびたバイト文字列リテラルはそうではないので、翻訳の中に1つを追加する必要があります。

Rustでは、長さ `n`のバイト文字列リテラルの型は`＆ 'staticです[u8; n] `。
Cのセマンティクスに合わせるために、代わりに生のポインタが必要です。
便利なことに、Rustスライスには生のポインタを抽出する `.as_ptr（）`メソッドがあります。
文字列リテラルは ''静的な '存続期間を持っているので、結果の生のポインタは常に安全に使用できます。

```haskell
    CStrConst (CString str False) _ -> return Result
        { resultType = IsArray Rust.Immutable (length str + 1) charType
        , resultMutable = Rust.Immutable
        , result = Rust.Deref (Rust.Lit (Rust.LitByteStr (str ++ "\NUL")))
        }
    _ -> unimplemented expr
    where
```

`42`のような数字はそれがどの型であるべきかについての情報を与えません。
Rustでは、数値リテラルにその型名を付けることができるので、 `42i8`は型` i8`を持ち、 `42f32`は型` f32`を持ちます。
`literalNumber`は、型のRust AST表現がこれらのリテラルの正しい接尾辞を取得するための単なる文字列であるという事実を悪用します。

Rustは接尾辞のない数字リテラルを許可します。その場合、周囲の文脈からどのような型の数字を推測しようとします。
しかし、ここではRustの推論規則に頼りたくはありません。代わりにCの規則と一致させる必要があるためです。

```haskell
    literalNumber ty lit = Result
        { resultType = ty
        , resultMutable = Rust.Immutable
        , result = Rust.Lit (lit (toRustType ty))
        }
```

C99複合リテラルは、初期化しているものの型が含まれている初期化子リストとそれほど違いはありません。

```haskell
interpretExpr _ (CCompoundLit decl initials info) = do
    (mut, ty) <- typeName decl
    final <- interpretInitializer ty (CInitList initials info)
    return Result
        { resultType = ty
        , resultMutable = mut
        , result = final
        }
```

GCCの「ステートメント式」拡張は、Rustブロック式にかなり直接的に変換されます。

```haskell
interpretExpr demand stat@(CStatExpr (CCompound [] stmts _) _) = scope $ do
    let (effects, final) = case last stmts of
            CBlockStmt (CExpr expr _) | demand -> (init stmts, expr)
            _ -> (stmts, Nothing)
    effects' <- cfgToRust stat (foldr interpretBlockItem (return ([], Unreachable)) effects)
    final' <- mapM (interpretExpr True) final
    return Result
        { resultType = maybe IsVoid resultType final'
        , resultMutable = maybe Rust.Immutable resultMutable final'
        , result = Rust.BlockExpr (Rust.Block effects' (fmap result final'))
        }
```

そうでなければ、このような式はまだ実装されていません。

```haskell
interpretExpr _ expr = unimplemented expr
```

> ** TODO **：これらの式ヘルパー関数を文書化してください。

```haskell
wrapping :: Result -> Result
wrapping r@(Result { resultType = IsInt Unsigned _ }) = case result r of
    Rust.Add lhs rhs -> r { result = Rust.MethodCall lhs (Rust.VarName "wrapping_add") [rhs] }
    Rust.Sub lhs rhs -> r { result = Rust.MethodCall lhs (Rust.VarName "wrapping_sub") [rhs] }
    Rust.Mul lhs rhs -> r { result = Rust.MethodCall lhs (Rust.VarName "wrapping_mul") [rhs] }
    Rust.Div lhs rhs -> r { result = Rust.MethodCall lhs (Rust.VarName "wrapping_div") [rhs] }
    Rust.Mod lhs rhs -> r { result = Rust.MethodCall lhs (Rust.VarName "wrapping_rem") [rhs] }
    Rust.Neg e -> r { result = Rust.MethodCall e (Rust.VarName "wrapping_neg") [] }
    _ -> r
wrapping r = r

toPtr :: Result -> Maybe Result
toPtr ptr@(Result { resultType = IsArray mut _ el }) = Just ptr
    { resultType = IsPtr mut el
    , result = castTo (IsPtr mut el) ptr
    }
toPtr ptr@(Result { resultType = IsPtr{} }) = Just ptr
toPtr _ = Nothing

binop :: CExpr -> CBinaryOp -> Result -> Result -> EnvMonad s Result
binop expr op lhs rhs = fmap wrapping $ case op of
    CMulOp -> promote expr Rust.Mul lhs rhs
    CDivOp -> promote expr Rust.Div lhs rhs
    CRmdOp -> promote expr Rust.Mod lhs rhs
    CAddOp -> case (toPtr lhs, toPtr rhs) of
        (Just ptr, _) -> return (offset ptr rhs)
        (_, Just ptr) -> return (offset ptr lhs)
        _ -> promote expr Rust.Add lhs rhs
        where
        offset ptr idx = ptr
            { result = Rust.MethodCall (result ptr) (Rust.VarName "offset") [castTo (IsInt Signed WordWidth) idx]
            }
    CSubOp -> case (toPtr lhs, toPtr rhs) of
        (Just lhs', Just rhs') -> do
            ptrTo <- case compatiblePtr (resultType lhs') (resultType rhs') of
                IsPtr _ ptrTo -> return ptrTo
                _ -> badSource expr "pointer subtraction of incompatible pointers"
            let ty = IsInt Signed WordWidth
            let size = rustSizeOfType (toRustType ptrTo)
            return Result
                { resultType = ty
                , resultMutable = Rust.Immutable
                , result = (Rust.MethodCall (castTo ty lhs') (Rust.VarName "wrapping_sub") [castTo ty rhs']) / castTo ty size
                }
        (Just ptr, _) -> return ptr { result = Rust.MethodCall (result ptr) (Rust.VarName "offset") [Rust.Neg (castTo (IsInt Signed WordWidth) rhs)] }
        _ -> promote expr Rust.Sub lhs rhs
    CShlOp -> shift Rust.ShiftL
    CShrOp -> shift Rust.ShiftR
    CLeOp -> comparison Rust.CmpLT
    CGrOp -> comparison Rust.CmpGT
    CLeqOp -> comparison Rust.CmpLE
    CGeqOp -> comparison Rust.CmpGE
    CEqOp -> comparison Rust.CmpEQ
    CNeqOp -> comparison Rust.CmpNE
    CAndOp -> promote expr Rust.And lhs rhs
    CXorOp -> promote expr Rust.Xor lhs rhs
    COrOp -> promote expr Rust.Or lhs rhs
    CLndOp -> return Result { resultType = IsBool, resultMutable = Rust.Immutable, result = Rust.LAnd (toBool lhs) (toBool rhs) }
    CLorOp -> return Result { resultType = IsBool, resultMutable = Rust.Immutable, result = Rust.LOr (toBool lhs) (toBool rhs) }
    where
    shift op' = return Result
        { resultType = lhsTy
        , resultMutable = Rust.Immutable
        , result = op' (castTo lhsTy lhs) (castTo rhsTy rhs)
        }
        where
        lhsTy = intPromote (resultType lhs)
        rhsTy = intPromote (resultType rhs)
    comparison op' = do
        res <- promotePtr expr op' lhs rhs
        return res { resultType = IsBool }
```

単項インクリメント演算子およびデクリメント演算子を含む代入式は、Cではかなり複雑な意味を持ちます。
まず、対応する二項演算子（ `x = x + ...`）を使って、複合代入（例えば `x + = ...`）を単純な代入に減らします。

```haskell
compound :: CExpr -> Bool -> Bool -> CAssignOp -> Result -> Result -> EnvMonad s Result
compound expr returnOld demand op lhs rhs = do
    let op' = case op of
            CAssignOp -> Nothing
            CMulAssOp -> Just CMulOp
            CDivAssOp -> Just CDivOp
            CRmdAssOp -> Just CRmdOp
            CAddAssOp -> Just CAddOp
            CSubAssOp -> Just CSubOp
            CShlAssOp -> Just CShlOp
            CShrAssOp -> Just CShrOp
            CAndAssOp -> Just CAndOp
            CXorAssOp -> Just CXorOp
            COrAssOp  -> Just COrOp
```

いくつかのC代入は、左側の式を複数回使用しなければならない複数のRustステートメントに変換する必要があります。
以下のいずれかの場合、左側を複製します。

1. Rust複合代入演算子は、C複合代入演算子が行うような多くのオペランド型の組み合わせをサポートしていないため、代入は複合代入です。
2.または周囲の式は代入の結果を使用します。その場合、Rust代入演算子は常に `（）`に評価されるため、L値を別々のステートメントで読み書きする必要があります。

```haskell
    let duplicateLHS = isJust op' || demand
```

ただし、左側の式に副作用がある可能性がある場合は、それらの効果を再現しないでください。
その場合は、もっと複雑な翻訳を使う必要があります。

- 左側を評価した結果の可変借用をバインドします。そうすることで、式を再度評価することなく、借用を複数回参照解除できます。

- しかし、右辺の式は左辺を捕らえるときに借りる変数を使うかもしれないので、右辺を最初に評価することを確実にしなければなりません。あるいはRustのボローチェッカーは文句を言うかもしれません。そのため、右側についてもlet-bindingを生成します。

一般的な場合に複雑な変換を使用しないようにするために、式に副作用がないことが保証されているかどうかを確認するために `hasNoSideEffects`（下記参照）を使用します。

```haskell
    let (bindings1, dereflhs, boundrhs) =
            if not duplicateLHS || hasNoSideEffects (result lhs)
            then ([], lhs, rhs)
            else
                let lhsvar = Rust.VarName "_lhs"
                    rhsvar = Rust.VarName "_rhs"
                in ([ Rust.Let Rust.Immutable rhsvar Nothing (Just (result rhs))
                    , Rust.Let Rust.Immutable lhsvar Nothing (Just (Rust.Borrow Rust.Mutable (result lhs)))
                    ], lhs { result = Rust.Deref (Rust.Var lhsvar) }, rhs { result = Rust.Var rhsvar })
    rhs' <- case op' of
        Just o -> binop expr o dereflhs boundrhs
        Nothing -> return boundrhs
    let assignment = Rust.Assign (result dereflhs) (Rust.:=) (castTo (resultType lhs) rhs')
```

次に、周囲の式が結果を使用しない限り、この代入が何を評価することになっているのかを把握する必要があります。
実際には、Cプログラムのほとんどの代入はその結果ではなく副作用についてのみ評価されるので、その場合を検出することによって生成するコードを大幅に単純化することができます。

しかし、結果が必要とされる数少ない代入については、結果が新しく代入された値（代入や前インクリメント/デクリメント演算子のように）なのか、古い値なのか（後インクリメント/デクリメント演算子のように） 。
後者の場合、代入を実行する前に古い値のコピーを束縛します。

```haskell
    let (bindings2, ret) =
            if not demand
            then ([], Nothing)
            else if not returnOld
            then ([], Just (result dereflhs))
            else
                let oldvar = Rust.VarName "_old"
                in ([Rust.Let Rust.Immutable oldvar Nothing (Just (result dereflhs))], Just (Rust.Var oldvar))
```

今度は生成されたlet束縛、代入文、および結果式をまとめ、それらをすべてまとめるためにブロック式が必要かどうかをチェックします。

結果がない場合（結果が使用されていないため）、 `void`型の結果を生成します。
さらに、let-bindingsがない場合は、ブロック式はまったく必要なく、代入式だけを返すことができます。

```haskell
    return $ case Rust.Block (bindings1 ++ bindings2 ++ exprToStatements assignment) ret of
        b@(Rust.Block body Nothing) -> Result
            { resultType = IsVoid
            , resultMutable = Rust.Immutable
            , result = case body of
                [Rust.Stmt e] -> e
                _ -> Rust.BlockExpr b
            }
        b -> lhs { result = Rust.BlockExpr b }
    where
```

式に副作用があるかどうかという問題に対する控えめな近似を計算します。
式に副作用がないことを確実に確認できる場合にのみ、hasNoSideEffectsはTrueを返しますが、それ以上の検査で複製しても安全な式であってもFalseを返すことがあります。

これも汎用の副作用チェッカーではありません。
これは、L値に表示される可能性のある式のみを処理します。これが、現在使用しているものすべてです。
それ以外の場合は、保守的な「False」の推測が返されます。

```haskell
    hasNoSideEffects (Rust.Var{}) = True
    hasNoSideEffects (Rust.Path{}) = True
    hasNoSideEffects (Rust.Member e _) = hasNoSideEffects e
    hasNoSideEffects (Rust.Deref p) = hasNoSideEffects p
    hasNoSideEffects _ = False
```

```haskell
rustSizeOfType :: Rust.Type -> Result
rustSizeOfType (Rust.TypeName ty) = Result
        { resultType = IsInt Unsigned WordWidth
        , resultMutable = Rust.Immutable
        , result = Rust.Call (Rust.Var (Rust.VarName ("::std::mem::size_of::<" ++ ty ++ ">"))) []
        }

rustAlignOfType :: Rust.Type -> Result
rustAlignOfType (Rust.TypeName ty) = Result
        { resultType = IsInt Unsigned WordWidth
        , resultMutable = Rust.Immutable
        , result = Rust.Call (Rust.Var (Rust.VarName ("::std::mem::align_of::<" ++ ty ++ ">"))) []
        }
```


定数式
--------------------

Cは、コンパイル時に評価できる式構文のサブセットを定義し、そのような定数式を配列サイズ式などのさまざまなものに使用できるようにします。

このセクションでは、定数式の実際の値を知る必要があるときにそれをRustに変換するために使用できる定数式エバリュエータを定義します。
これは必ずしも必要なわけではありません。多くの場合、同等の定数式をRustで使用できます。そうすることで、プログラマの意図をより多く維持できるため、この方法をお勧めします。

```haskell
interpretConstExpr :: CExpr -> EnvMonad s Integer
interpretConstExpr (CConst (CIntConst (CInteger v _ _) _)) = return v
interpretConstExpr expr = unimplemented expr
```


暗黙の型強制
=======================

Cは暗黙的にさまざまな型変換を実行します。
Rustには、指示しなくても型間の変換が行われるケースがはるかに少なくなります。
そのため、Cの暗黙的な強制をRustの明示的なキャストとしてエンコードする必要があります。

一方、翻訳された式がすでに目的の型を持っている場合は、生成されたソースを乱雑にするようなキャスト式を発行したくはありません。
そのため `castTo`はキャストが必要な場合にのみキャストを挿入するスマートコンストラクタです。

```haskell
castTo :: CType -> Result -> Rust.Expr
castTo target source | resultType source == target = result source
```

配列型からC互換のキャストを生成する前に、まず生のポインタに変換する必要があります。
それから私達はそれを希望の型にキャストしようと試みることができます。

```haskell
castTo target (Result { resultType = IsArray mut _ el, result = source }) =
    castTo target Result
        { resultType = IsPtr mut el
        , resultMutable = Rust.Immutable
        , result = Rust.MethodCall source (Rust.VarName method) []
        }
    where
    method = case mut of
        Rust.Immutable -> "as_ptr"
        Rust.Mutable -> "as_mut_ptr"
```

Rustは他の型を直接 `bool`にキャストすることを許可していないので、代わりに以下で定義されている特別な場合のブール変換を使用してください。

```haskell
castTo IsBool source = toBool source
```

Cの整数リテラルが定義されている方法のために、リテラルを `i32`として翻訳し、それから実際にそれを` u8`かそれ以外にしたいということを発見することは非常に一般的です。
`42i32 as（u8）`を単に `42u8`に単純化することは生成されたコードを読みやすくそして維持しやすくするだけでなく、Rustコンパイラがそれらの型の法的範囲外の有用な警告を与えることも可能にします。

ただし、これには注意点が1つあります。
Rustでは、符号なし型に対して負の整数リテラルを使用できません。
警告するだけではありません。実行しようとすると、コンパイル時エラーが発生します。
欲しい結果は、そのような負の値を大きな正の値として囲むことです。
符号付き整数リテラルを発行してからunsignedにキャストすると、正しい結果が得られます。これは、単純化せずに行ったことなので、この場合は単にこの規則を抑制できます。

```haskell
castTo target@(IsInt{}) (Result { result = Rust.Lit (Rust.LitInt n repr _) })
    = Rust.Lit (Rust.LitInt n repr (toRustType target))
castTo (IsInt Signed w) (Result { result = Rust.Neg (Rust.Lit (Rust.LitInt n repr _)) })
    = Rust.Neg (Rust.Lit (Rust.LitInt n repr (toRustType (IsInt Signed w))))
```

特別な場合が当てはまらない場合は、Rustキャスト式を発行してください。

```haskell
castTo target source = Rust.Cast (result source) (toRustType target)
```

同様に、Rustが `bool`型の式を必要とする場合、またはCが式がそれぞれ" false "と" true "を表す" 0 "または" not 0 "のいずれかとして解釈されることを必要とする場合私たちが持っている任意のスカラー式を `bool`に変換します。

```haskell
toBool :: Result -> Rust.Expr
```

整数リテラル式を `bool`に変換するためには、型を見る必要はありません。値だけです。
我々は特定の値1と0のみを変換します。なぜならそれらはCにおける `true`と` false`の慣用表現であるからです。
他の値をブール値のコンテキストで使用すると驚くことになります。開発者の注意を引くために、それらをより冗長に変換することを選択します。

```haskell
toBool (Result { result = Rust.Lit (Rust.LitInt 0 _ _) })
    = Rust.Lit (Rust.LitBool False)
toBool (Result { result = Rust.Lit (Rust.LitInt 1 _ _) })
    = Rust.Lit (Rust.LitBool True)
toBool (Result { resultType = t, result = v }) = case t of
```

式はすでにブール型を持っている可能性があります。その場合は変更せずに返すことができます。

```haskell
    IsBool -> v
```

あるいは、それはポインタかもしれません、その場合我々はそれがnullポインタではないかどうかに関するテストを生成する必要があります。

```haskell
    IsPtr _ _ -> Rust.Not (Rust.MethodCall v (Rust.VarName "is_null") [])
```

そうでなければ、それは数値でなければならず、それが0に等しくないことをテストする必要があります。

```haskell
    _ -> Rust.CmpNE v 0
```

toNotBoolはRust.Notと同じように動作します。
より単純な式を生成できることを除いて、toBool`。
例えば、 `!! p.is_null（）`の代わりに、単に `p.is_null（）`を生成します。
このアプローチは、構造的に可能な限り入力に近いRustを生成するという私たちの設計目標を満たします。

```haskell
toNotBool :: Result -> Rust.Expr
toNotBool (Result { result = Rust.Lit (Rust.LitInt 0 _ _) })
    = Rust.Lit (Rust.LitBool True)
toNotBool (Result { result = Rust.Lit (Rust.LitInt 1 _ _) })
    = Rust.Lit (Rust.LitBool False)
toNotBool (Result { resultType = t, result = v }) = case t of
    IsBool -> Rust.Not v
    IsPtr _ _ -> Rust.MethodCall v (Rust.VarName "is_null") []
    _ -> Rust.CmpEQ v 0
```

Cは、「整数プロモーション」と呼ばれる一連の規則を定義しています（C99セクション6.3.1.1段落2）。
`intPromote`はこれらの規則を型変換にエンコードします。
式をその整数で促進される型に変換するには、with `castTo`を使います。

```haskell
intPromote :: CType -> CType
```

仕様から、「intが元の型のすべての値を表すことができる場合、値はintに変換されます。」

ここでは `int`が32ビット幅であるとします。
この仮定は現代の主要なCPUアーキテクチャにも当てはまります。
また、特定のコンパイラの振る舞いを複製するのではなく、Cの標準準拠の実装を作成しようとしています。C標準では、intを任意のサイズに定義できます。
とはいえ、この仮定は移植性のないCプログラムを壊すかもしれません。

便利なことに、Rustは `bool`と` enum`型の式を任意の整数型にキャストすることを可能にし、Cが要求するように `true`を1に、` false`を0に変換します。ブール値または列挙型をここで処理します。

```haskell
intPromote IsBool = IsInt Signed (BitWidth 32)
intPromote (IsEnum _) = enumReprType
intPromote (IsInt _ (BitWidth w)) | w < 32 = IsInt Signed (BitWidth 32)
```

「それ以外の場合は、unsigned intに変換されます。他のすべての型は整数の昇格によって変更されません。」

```haskell
intPromote x = x
```

Cはまた、どんな型の二項演算子が評価されるべきかを決定するために "通常の算術変換"（C99セクション6.3.1.8）と呼ばれる規則のセットを定義します。

```haskell
usual :: CType -> CType -> Maybe CType
usual (IsFloat aw) (IsFloat bw) = Just (IsFloat (max aw bw))
usual a@(IsFloat _) _ = Just a
usual _ b@(IsFloat _) = Just b
```

「それ以外の場合、整数の昇格は両方のオペランドで実行されます。」

```haskell
usual origA origB = case (intPromote origA, intPromote origB) of
```

「次に、昇格したオペランドに次の規則が適用されます。」

昇格された型が整数型ではない場合、または整数変換ランクが `integerConversionRank`に準拠していない場合、この式の変換を拒否することに注意してください。
後者は、整数変換ランクの総合的な順序を記述しているC標準の何にも起因していません。
そうではなく、移植性のないコードで翻訳が複雑になるため、これらのプログラムの翻訳は拒否しています。

" - "両方のオペランドの型が同じであれば、それ以上の変換は不要です。

    ```haskell
        (a, b) | a == b -> Just a
    ```

" - それ以外の場合、両方のオペランドが符号付き整数型を持つ場合、または両方が符号なし整数型を持つ場合は、整数変換ランクが小さい方のオペランドがランクが大きいオペランドの型に変換されます。

    ```haskell
        (IsInt Signed sw, IsInt Unsigned uw) -> mixedSign sw uw
        (IsInt Unsigned uw, IsInt Signed sw) -> mixedSign sw uw
        (IsInt as aw, IsInt _bs bw) -> do
            rank <- integerConversionRank aw bw
            Just (IsInt as (if rank == GT then aw else bw))
        _ -> Nothing
        where
    ```

この時点で、符号付きオペランドと符号なしオペランドが1つずつあります。
通常の算術変換では、オペランドの順序を気にする必要はありません。そのため、mixedSignは符号付き幅が常に最初の引数で符号なし幅が常に2番目のヘルパー関数です。

" - それ以外の場合、符号なし整数型を持つオペランドのランクが他のオペランドの型のランク以上である場合、符号付き整数型のオペランドは符号なし整数型のオペランドの型に変換されます。

    ```haskell
        mixedSign sw uw = do
            rank <- integerConversionRank uw sw
            Just $ case rank of
                GT -> IsInt Unsigned uw
                EQ -> IsInt Unsigned uw
    ```

 -  "それ以外の場合、符号付き整数型のオペランドの型が符号なし整数型のオペランドの型のすべての値を表すことができる場合、符号なし整数型のオペランドは符号付き整数型のオペランドの型に変換されます。 "

    符号なし型のビット幅が符号付き型のビット幅より厳密に小さい場合、符号付き型は符号なし型のすべての値を表すことができます。

    我々の目的のために、 `long`は` int`よりも小さい `unsigned`型の値しか表現できません（` long`と `int`は同じサイズかもしれないので）。
    しかし、 `unsigned long`値は64ビットより大きい符号付き型でしか表現できません（` long`は代わりに64ビットかもしれないので）。

    ```haskell
                _ | bitWidth 64 uw < bitWidth 32 sw -> IsInt Signed sw
    ```

" - それ以外の場合、両方のオペランドは符号付き整数型を持つオペランドの型に対応する符号なし整数型に変換されます。"

    上記の定義を考えると、これは `unsigned long`と` int64_t`に対してのみ起こります。ここでは `uint64_t`を選びます。

    ```haskell
                _ -> IsInt Unsigned sw
    ```

通常の算術変換は、C99 6.3.1.1から "整数変換ランク"と呼ばれる定義を参照しています。
これらの幅が厳密に増加する整数変換ランクを持つようにします。

 -  `BitWidth 32`（` int`の表現）
 -  `WordWidth`（` long`の表現）
 -  `BitWidth 64`

最初の2つはC99 6.3.1.1から来たもので、「long」は「int」よりも高いランクを持っています。
最後のものを実装の選択肢として追加します。
`isize`は` i32`でも `i64`でもかまいませんので、` isize`を他の型と組み合わせるときは、どちらか大きいほうの型にぶつかるでしょう。

32ビットから64ビットまでのビット幅を、ワードサイズと比較して比較することはできません。
ワードサイズはプラットフォームに依存するため、どちらが大きいかわからない。

```haskell
integerConversionRank :: IntWidth -> IntWidth -> Maybe Ordering
integerConversionRank (BitWidth a) (BitWidth b) = Just (compare a b)
integerConversionRank WordWidth WordWidth = Just EQ
integerConversionRank (BitWidth a) WordWidth
    | a <= 32 = Just LT
    | a >= 64 = Just GT
integerConversionRank WordWidth (BitWidth b)
    | b <= 32 = Just GT
    | b >= 64 = Just LT
integerConversionRank _ _ = Nothing
```

これは、通常の算術変換を二項演算子の両方のオペランドに適用し、必要に応じてキャストしてから、任意のRust二項演算子を使用してそれらのオペランドを結合するためのヘルパー関数です。

```haskell
promote
    :: (Pretty node, Pos node)
    => node
    -> (Rust.Expr -> Rust.Expr -> Rust.Expr)
    -> Result -> Result -> EnvMonad s Result
promote node op a b = case usual (resultType a) (resultType b) of
    Just rt -> return Result
        { resultType = rt
        , resultMutable = Rust.Immutable
        , result = op (castTo rt a) (castTo rt b)
        }
    Nothing -> badSource node $ concat
        [ "arithmetic combination for "
        , show (resultType a)
        , " and "
        , show (resultType b)
        ]
```

`compatiblePtr`は"通常の算術変換 "と同等のものを実装していますが、ポインタ型用です。

> ** FIXME **：C99への引用が必要です。

```haskell
compatiblePtr :: CType -> CType -> CType
compatiblePtr (IsPtr _ IsVoid) b = b
compatiblePtr (IsArray mut _ el) b = compatiblePtr (IsPtr mut el) b
compatiblePtr a (IsPtr _ IsVoid) = a
compatiblePtr a (IsArray mut _ el) = compatiblePtr a (IsPtr mut el)
compatiblePtr (IsPtr m1 a) (IsPtr m2 b) = IsPtr (leastMutable m1 m2) (compatiblePtr a b)
    where
    leastMutable Rust.Mutable Rust.Mutable = Rust.Mutable
    leastMutable _ _ = Rust.Immutable
compatiblePtr a b | a == b = a
```

ここに来たら、ポインタ型は互換性がありません、私が言うことができる限りC99によって許されない限り。
しかしGCCはそれを警告として扱うだけなので、両側をvoidポインタにキャストします。これは通常のアーキテクチャで機能するはずです。

```haskell
compatiblePtr _ _ = IsVoid
```

最後に、 `promPtr`は`昇格 `に似ていますが、算術型だけではなく、ポインタとしてオペランドを許す演算子のためのものです。
整数リテラルは暗黙のうちにポインタとして使用されるかもしれないので、どちらかのオペランドがポインタの場合、もう一方のオペランドはvoidポインタのふりをして、それが実際にどの型に変換されるべきかを `compatiblePtr`に考えさせます。

```haskell
promotePtr
    :: (Pretty node, Pos node)
    => node
    -> (Rust.Expr -> Rust.Expr -> Rust.Expr)
    -> Result -> Result -> EnvMonad s Result
promotePtr node op a b = case (resultType a, resultType b) of
    (IsArray _ _ _, _) -> ptrs
    (IsPtr _ _, _) -> ptrs
    (_, IsArray _ _ _) -> ptrs
    (_, IsPtr _ _) -> ptrs
    _ -> promote node op a b
    where
    ptrOrVoid r = case resultType r of
        t@(IsArray _ _ _) -> t
        t@(IsPtr _ _) -> t
        _ -> IsPtr Rust.Mutable IsVoid
    ty = compatiblePtr (ptrOrVoid a) (ptrOrVoid b)
    ptrs = return Result
        { resultType = ty
        , resultMutable = Rust.Immutable
        , result = op (castTo ty a) (castTo ty b)
        }
```


C型の内部表現
=================================

> ** TODO **：ここで使用している型表現を文書化してください。

```haskell
data Signed = Signed | Unsigned
    deriving (Show, Eq)

data IntWidth = BitWidth Int | WordWidth
    deriving (Show, Eq)
```

時には `WordWidth`を特定のビット数に相当するものとして扱いたいと思うかもしれませんが、その選択はそれを最小の幅にするか最大の幅にするかによって異なります。
（私たちはマシンのワードサイズを常に32か64のどちらかとして定義します。なぜならそれらは現在Rustがターゲットとしている唯一のサイズだからです。）

```haskell
bitWidth :: Int -> IntWidth -> Int
bitWidth wordWidth WordWidth = wordWidth
bitWidth _ (BitWidth w) = w
```

```haskell
data CType
    = IsBool
    | IsInt Signed IntWidth
    | IsFloat Int
    | IsVoid
    | IsFunc CType [(Maybe (Rust.Mutable, Ident), CType)] Bool
    | IsPtr Rust.Mutable CType
    | IsArray Rust.Mutable Int CType
    | IsStruct String [(String, CType)]
    | IsEnum String
    | IsIncomplete Ident
    deriving Show
```

同等性のために `Eq`型クラスのデフォルト実装を導出することは、引数名と` const`-nessが型を異ならずに異なる可能性がある関数型の表現を除いて、ほとんどうまくいきます。

```haskell
instance Eq CType where
    IsBool == IsBool = True
    IsInt as aw == IsInt bs bw = as == bs && aw == bw
    IsFloat aw == IsFloat bw = aw == bw
    IsVoid == IsVoid = True
    IsFunc aRetTy aFormals aVariadic == IsFunc bRetTy bFormals bVariadic =
        aRetTy == bRetTy && aVariadic == bVariadic &&
        map snd aFormals == map snd bFormals
    IsPtr aMut aTy == IsPtr bMut bTy = aMut == bMut && aTy == bTy
    IsArray aMut _ aTy == IsArray bMut _ bTy = aMut == bMut && aTy == bTy
    IsStruct aName aFields == IsStruct bName bFields =
        aName == bName && aFields == bFields
    IsEnum aName == IsEnum bName = aName == bName
    IsIncomplete aName == IsIncomplete bName = aName == bName
    _ == _ = False
```

```haskell
toRustType :: CType -> Rust.Type
toRustType IsBool = Rust.TypeName "bool"
toRustType (IsInt s w) = Rust.TypeName ((case s of Signed -> 'i'; Unsigned -> 'u') : (case w of BitWidth b -> show b; WordWidth -> "size"))
toRustType (IsFloat w) = Rust.TypeName ('f' : show w)
toRustType IsVoid = Rust.TypeName "::std::os::raw::c_void"
toRustType (IsFunc retTy args variadic) = Rust.TypeName $ concat
    [ "unsafe extern fn("
    , args'
    , ")"
    , if retTy /= IsVoid then " -> " ++ typename retTy else ""
    ]
    where
    typename (toRustType -> Rust.TypeName t) = t
    args' = intercalate ", " (
            map (typename . snd) args ++ if variadic then ["..."] else []
        )
toRustType (IsPtr mut to) = let Rust.TypeName to' = toRustType to in Rust.TypeName (rustMut mut ++ to')
    where
    rustMut Rust.Mutable = "*mut "
    rustMut Rust.Immutable = "*const "
toRustType (IsArray _ size el) = Rust.TypeName ("[" ++ typename el ++ "; " ++ show size ++ "]")
    where
    typename (toRustType -> Rust.TypeName t) = t
toRustType (IsStruct name _fields) = Rust.TypeName name
toRustType (IsEnum name) = Rust.TypeName name
toRustType (IsIncomplete ident) = Rust.TypeName (identToString ident)
```

何も返さない関数は、Rustの戻り値の型が他のCの `void`型の表現とは異なります。

```haskell
toRustRetType :: CType -> Rust.Type
toRustRetType IsVoid = Rust.TypeName "()"
toRustRetType ty = toRustType ty
```

Cは基本のchar型が符号付きか符号なしかを決定するために実装に任せます。
これはRustのバイトリテラルの一種なので、符号なしを選択します。

```haskell
charType :: CType
charType = IsInt Unsigned (BitWidth 8)
```

Cは実装が `int`より狭い整数型を使用してenum値を表現することを可能にします、しかし今のところ我々はそうしないことを選択します。
これはいくつかのABIと互換性がないかもしれません。

```haskell
enumReprType :: CType
enumReprType = IsInt Signed (BitWidth 32)
```

`CType`は対応するRust型に簡単に変換できるように簡略化されたC型の表現です - しかし生のC ASTから` CType`を構築する過程で、ここで私たちが追跡する必要があるいくつかの余分な状態があります。

```haskell
data IntermediateType = IntermediateType
    { typeMutable :: Rust.Mutable
    , typeIsFunc :: Bool
    , typeRep :: CType
    }
```

型や `extern`宣言のような特定のCの構成要素を翻訳するとき、宣言が実際に使われるのがわかるまで翻訳を延期します。
しかし、私たちは物事を何度も翻訳したくはありません。
そのため `runOnce`は、後で実行するための何らかのアクションを与えられて、（Rustの` Cell`や `RefCell`型のような）可変の参照セルにまとめられます。
次に、新しいアクションを返します。これは、参照セルの現在の内容を読み取り、必要に応じて元のアクションを実行して、結果を参照セルにキャッシュします。

```haskell
runOnce :: EnvMonad s a -> EnvMonad s (EnvMonad s a)
runOnce action = do
    cacheRef <- lift $ lift $ newSTRef (Left action)
    return $ do
        cache <- lift $ lift $ readSTRef cacheRef
        case cache of
            Left todo -> do
                lift $ lift $ writeSTRef cacheRef $ Left $
                    fail "internal error: runOnce action depends on itself, leading to an infinite loop"
                val <- todo
                lift $ lift $ writeSTRef cacheRef (Right val)
                return val
            Right val -> return val
```

`static`、` const`、または `int`のような宣言指定子の袋を与えられて、記述された型の私達自身の表現を構築します。
その過程で、新しい `struct`、` union`、または `enum`型のネストした宣言を環境に追加します。

```haskell
baseTypeOf :: [CDeclSpec] -> EnvMonad s (Maybe CStorageSpec, EnvMonad s IntermediateType)
baseTypeOf specs = do
    -- TODO：プロセス属性と `inline`キーワード
    let (storage, _attributes, basequals, basespecs, _inlineNoReturn, _align) = partitionDeclSpecs specs
    mstorage <- case storage of
        [] -> return Nothing
        [spec] -> return (Just spec)
        _ : excess : _ -> badSource excess "extra storage class specifier"
    base <- typedef (mutable basequals) basespecs
    return (mstorage, base)
    where
```

型指定子が `typedef`への参照を含む場合、それが唯一の型指定子でなければなりません。

```haskell
    typedef mut [spec@(CTypeDef ident _)] = do
```

`typedef`を使うのはややこしいです、なぜならそれらは` const`を焼き付けたり、関数ポインタではなく関数型を表現することができるからです。
`typedef`またはこの指定子リストの少なくとも1つが` const`を含んでいれば、結果の型は `const`です。
他の型情報は `typedef`からコピーされたばかりです。

```haskell
        (name, mty) <- getTypedefIdent ident
        case mty of
            Just deferred | mut == Rust.Immutable ->
                return (fmap (\ itype -> itype { typeMutable = Rust.Immutable }) deferred)
            Just deferred -> return deferred
```

GCCヘッダは `va_list`を実装する型として` __builtin_va_list`を使います。
私がチェックしたすべてのプラットフォームで、この型のABIはポインタと互換性があるので、それを一意の不完全型へのポインタに変換します。

```haskell
            Nothing | name == "__builtin_va_list" -> runOnce $ do
                ty <- emitIncomplete Type ident
                return IntermediateType
                    { typeMutable = mut
                    , typeIsFunc = False
                    , typeRep = IsPtr Rust.Mutable ty
                    }
            Nothing -> badSource spec "undefined type"
```

他の型指定子は型を `const`にすることができないので、残りの場合は` CType`を返すようにしましょう。そしてそれを共通の追加フィールドでまとめます。

```haskell
    typedef mut other = do
        deferred <- singleSpec other
        return (fmap (simple mut) deferred)

    simple mut ty = IntermediateType
        { typeMutable = mut
        , typeIsFunc = False
        , typeRep = ty
        }
```

`typedef`型のように、この型指定子の次のグループはどれも他の型指定子と一緒に使うことはできません。

```haskell
    singleSpec [CVoidType _] = return (return IsVoid)
    singleSpec [CBoolType _] = return (return IsBool)
```

フィールドを持たない `struct`はタグを持たなければならず、フィールドが他の場所に与えられている` struct`への参照を表します。
これが前方宣言である場合、実際の宣言が処理されるまでこの型の値を構築またはアクセスすることはできません。そのため、それまではこの型を不完全としてマークします。

```haskell
    singleSpec [CSUType (CStruct CStructTag (Just ident) Nothing _ _) _] = do
        mty <- getTagIdent ident
        return $ fromMaybe (emitIncomplete Struct ident) mty
```

`struct`宣言を翻訳することは各フィールドの型を再帰的に翻訳することから始まります。
後で定義が現在スコープ内にあるものを隠す前に、これらの型を調べる必要がありますが、この構造体自体が使用されていない限り、これらの型を使用済みとして扱わないように注意する必要があります。
そのため、後で据え置き型宣言を保存します。

ビットフィールドはまだ翻訳されていませんが、この構造体が実際に使用されている場合にのみエラーを報告するので、私たちはそれらについても怠惰になることができます。

```haskell
    singleSpec [CSUType (CStruct CStructTag mident (Just declarations) _ _) _] = do
        deferredFields <- fmap concat $ forM declarations $ \ declaration -> case declaration of
          CStaticAssert {} -> return []
          CDecl spec decls _ -> do
            (storage, base) <- baseTypeOf spec
            case storage of
                Just s -> badSource s "storage class specifier in struct"
                Nothing -> return ()
            forM decls $ \ decl -> case decl of
                (Just declr@(CDeclr (Just field) _ _ _ _), Nothing, Nothing) -> do
                    deferred <- derivedDeferredTypeOf base declr []
                    return (applyRenames field, deferred)
                (_, Nothing, Just _size) -> do
                    return ("<bitfield>", unimplemented declaration)
                _ -> badSource declaration "field in struct"
```

次のステップはこの `struct`が使われている場合にのみ起こり、せいぜい一度だけ起こるべきです。

```haskell
        deferred <- runOnce $ do
```

Cでは、 `struct`は匿名にすることができます。
さびではありません。
私たちが匿名の `struct`に遭遇したら、それにユニークな名前を作りそれを使う必要があります。

```haskell
            (shouldEmit, name) <- case mident of
                Just ident -> do
                    rewrites <- lift (asks itemRewrites)
                    case Map.lookup (Struct, identToString ident) rewrites of
                        Just renamed -> return (False, concatMap ("::" ++) renamed)
                        Nothing -> return (True, identToString ident)
                Nothing -> do
                    name <- uniqueName "Struct"
                    return (True, name)
```

それでは、この `struct`のフィールドに必要なすべての型を翻訳しましょう。

```haskell
            fields <- forM deferredFields $ \ (fieldName, deferred) -> do
                itype <- deferred
                return (fieldName, typeRep itype)
```

Cはお互いに `struct`変数を代入し、それらのコピーを値で関数呼び出しに渡すことを可能にします。
Rustで同じ振る舞いをさせるためには、 `Copy`特性の実装が必要です、そしてそれは同様に` Clone`を必要とします。
通常、これらの両方の特性を自動導出しますが、残念ながら32より大きいサイズの配列は `クローン`を実装していません。
しかしながら、それらは常に（その要素がコピーされることができる限り） `Copy`を実装するので、単に構造体をコピーする明示的な` Clone`実装を生成します。

また、Rustコンパイラがターゲットプラットフォーム用のC ABIと同じ規則を使ってこの `struct`のフィールドをレイアウトすることを要求します。

```haskell
            let attrs = [Rust.Attribute "derive(Copy)", Rust.Attribute "repr(C)"]
            when shouldEmit $ emitItems
                [ Rust.Item attrs Rust.Public (Rust.Struct name [ (field, toRustType fieldTy) | (field, fieldTy) <- fields ])
                , Rust.Item [] Rust.Private (Rust.CloneImpl (Rust.TypeName name))
                ]
            return (IsStruct name fields)
```

この時点で、このタイプが使用された場合に後で実行することになるすべてのアクションを取っておきます。
今度は、次のいずれかの場合にこれらのアクションが確実に実行されるようにする必要があります。またはこの型が出現した宣言もいくつかのシンボルを宣言しています。

```haskell
        case mident of
            Just ident -> addTagIdent ident deferred
            Nothing -> return ()
        return deferred
```

これを書いている時点で、Cスタイルの `union`型に対するRustのサポートはつい最近、毎晩rustcに上陸しました。
それが安定するまで、Cの `共用体`型を完全に翻訳することはできません。
しかし、私たちにできることは、そのような共用体へのポインタをどこにでも渡すことができるようにすることです。
だから今のところ、これはそれぞれの `共用体`に対して不完全な型を作ります。

```haskell
    singleSpec [CSUType (CStruct CUnionTag mident _ _ _) node] = runOnce $ do
        ident <- case mident of
            Just ident -> return ident
            Nothing -> do
                name <- uniqueName "Union"
                return (internalIdentAt (posOfNode node) name)
        emitIncomplete Union ident
```

`struct`参照とは異なり、` enum`参照は既に宣言されている `enum`を指定しなければなりません。

```haskell
    singleSpec [spec@(CEnumType (CEnum (Just ident) Nothing _ _) _)] = do
        mty <- getTagIdent ident
        case mty of
            Just ty -> return ty
            Nothing -> badSource spec "undefined enum"
```

Cでは、 `enum`宣言は新しい型を作成するだけでなく、シンボル環境に追加する定数のコレクションも宣言します。

```haskell
    singleSpec [CEnumType (CEnum mident (Just items) _ _) _] = do
        deferred <- runOnce $ do
            (shouldEmit, name) <- case mident of
                Just ident -> do
                    rewrites <- lift (asks itemRewrites)
                    case Map.lookup (Enum, identToString ident) rewrites of
                        Just renamed -> return (False, concatMap ("::" ++) renamed)
                        Nothing -> return (True, identToString ident)
                Nothing -> do
                    name <- uniqueName "Enum"
                    return (True, name)

            -- FIXME: these expressions should be evaluated in the
            -- environment from declaration time, not at first use.
            enums <- forM items $ \ (ident, mexpr) -> do
                let enumName = applyRenames ident
                case mexpr of
                    Nothing -> return (Rust.EnumeratorAuto enumName)
                    Just expr -> do
                        expr' <- interpretExpr True expr
                        return (Rust.EnumeratorExpr enumName (castTo enumReprType expr'))
```

上記の `struct`のように、` enum`は `Copy`と` Clone`の両方を必要とします。
しかし、それぞれの `enum`も上で定義した` enumReprType`のように表現されるように強制します。

```haskell
            let Rust.TypeName repr = toRustType enumReprType
            let attrs = [ Rust.Attribute "derive(Clone, Copy)"
                        , Rust.Attribute (concat [ "repr(", repr, ")" ])
                        ]
            when shouldEmit $
                emitItems [Rust.Item attrs Rust.Public (Rust.Enum name enums)]
            return (IsEnum name)

        forM_ items $ \ (ident, _mexpr) -> addSymbolIdentAction ident $ do
            IsEnum name <- deferred
            return Result
                { resultType = IsEnum name
                , resultMutable = Rust.Immutable
                , result = Rust.Path (Rust.PathSegments [name, applyRenames ident])
                }

        case mident of
            Just ident -> addTagIdent ident deferred
            Nothing -> return ()
        return deferred
```

もしどれも型指定子リストにマッチしなければ、うまくいけばそれは算術型です。
Cのデフォルトの型は `signed int`なので、そこから始めて、どんな型指定子があってもそれを修正します。

```haskell
    singleSpec other = return (foldrM arithmetic (IsInt Signed (BitWidth 32)) other)

    arithmetic :: CTypeSpec -> CType -> EnvMonad s CType
    arithmetic (CSignedType _) (IsInt _ width) = return (IsInt Signed width)
    arithmetic (CUnsigType _) (IsInt _ width) = return (IsInt Unsigned width)
    arithmetic (CCharType _) _ = return charType
    arithmetic (CShortType _) (IsInt s _) = return (IsInt s (BitWidth 16))
    arithmetic (CIntType _) (IsInt s _) = return (IsInt s (BitWidth 32))
    arithmetic (CLongType _) (IsInt s _) = return (IsInt s WordWidth)
    arithmetic (CLongType _) (IsFloat w) = return (IsFloat w)
    arithmetic (CFloatType _) _ = return (IsFloat 32)
    arithmetic (CDoubleType _) _ = return (IsFloat 64)
    arithmetic spec _ = unimplemented spec
```

単純型の計算方法を説明したばかりですが、C宣言子によって複雑さが増します。
宣言子は、単純型をゼロ個以上の派生型で帰納的に囲むことができます。各派生型は、ポインタ、配列、または関数型です。

```haskell
derivedTypeOf :: EnvMonad s IntermediateType -> CDeclr -> EnvMonad s IntermediateType
derivedTypeOf deferred declr = join (derivedDeferredTypeOf deferred declr [])

derivedDeferredTypeOf
    :: EnvMonad s IntermediateType
    -> CDeclr
    -> [CDecl]
    -> EnvMonad s (EnvMonad s IntermediateType)
derivedDeferredTypeOf deferred declr@(CDeclr _ derived _ _ _) argtypes = do
    derived' <- mapM derive derived
    return $ do
        basetype <- deferred
        foldrM ($) basetype derived'
    where
```

内部型表現では、 `IsFunc`は関数_pointer_です。
したがって、 `CPtrDeclr`の後に` CFunDeclr`が続くのを見れば、ポインタを食べるべきです。

`CFunDeclr`の前に` CArrDeclr`または `CFunDeclr`があると、それはエラーです。介在するポインタがなければなりません。

```haskell
    derive (CPtrDeclr quals _) = return $ \ itype ->
        if typeIsFunc itype
        then return itype { typeIsFunc = False }
        else return itype
            { typeMutable = mutable quals
            , typeRep = IsPtr (typeMutable itype) (typeRep itype)
            }
    derive (CArrDeclr quals arraySize _) = return $ \ itype ->
        if typeIsFunc itype
        then badSource declr "function as array element type"
        else do
            sizeExpr <- case arraySize of
                CArrSize _ sizeExpr -> return sizeExpr
                CNoArrSize _ -> unimplemented declr
            size <- interpretConstExpr sizeExpr
            return itype
                { typeMutable = mutable quals
                , typeRep = IsArray (typeMutable itype) (fromInteger size) (typeRep itype)
                }
```

```haskell
    derive (CFunDeclr foo _ _) = do
        let preAnsiArgs = Map.fromList
                [ (argname, CDecl argspecs [(Just declr', Nothing, Nothing)] pos)
                | CDecl argspecs declrs _ <- argtypes
                , (Just declr'@(CDeclr (Just argname) _ _ _ pos), Nothing, Nothing) <- declrs
                ]
        (args, variadic) <- case foo of
            Right (args, variadic) -> return (args, variadic)
            Left argnames -> do
                argdecls <- forM argnames $ \ argname ->
                    case Map.lookup argname preAnsiArgs of
                    Nothing -> badSource declr ("undeclared argument " ++ show (identToString argname))
                    Just arg -> return arg
                return (argdecls, False)
        args' <- sequence
            [ do
                (storage, base') <- baseTypeOf argspecs
                case storage of
                    Nothing -> return ()
                    Just (CRegister _) -> return ()
                    Just s -> badSource s "storage class specifier on argument"
                (argname, argTy) <- case declr' of
                    [] -> return (Nothing, base')
                    [(Just argdeclr@(CDeclr argname _ _ _ _), Nothing, Nothing)] -> do
                        argTy <- derivedDeferredTypeOf base' argdeclr []
                        return (argname, argTy)
                    _ -> badSource arg "function argument"
                return $ do
                    itype <- argTy
                    when (typeIsFunc itype) (badSource arg "function as function argument")
                    let ty = case typeRep itype of
                            IsArray mut _ el -> IsPtr mut el
                            orig -> orig
                    return (fmap ((,) (typeMutable itype)) argname, ty)
            | arg@(CDecl argspecs declr' _) <-
```

引数リスト `（void）`と `（）`を同じものとして扱います。両方とも関数が引数を取らないことを意味します。

```haskell
                case args of
                [CDecl [CTypeSpec (CVoidType _)] [] _] -> []
                _ -> args
            ]
        return $ \ itype -> do
            when (typeIsFunc itype) (badSource declr "function as function return type")
            args'' <- sequence args'
            return itype
                { typeIsFunc = True
                , typeRep = IsFunc (typeRep itype) args'' variadic
                }
```

上記のいくつかの場所は型修飾子が `const`であるかどうかをチェックする必要があるので、このヘルパー関数はその質問に答えます。

```haskell
mutable :: [CTypeQualifier a] -> Rust.Mutable
mutable quals = if any (\ q -> case q of CConstQual _ -> True; _ -> False) quals then Rust.Immutable else Rust.Mutable

typeName :: CDecl -> EnvMonad s (Rust.Mutable, CType)
typeName decl@(CStaticAssert {}) = badSource decl "static assert in type name "
typeName decl@(CDecl spec declarators _) = do
    (storage, base) <- baseTypeOf spec
    case storage of
        Just s -> badSource s "storage class specifier in type name"
        Nothing -> return ()
    itype <- case declarators of
        [] -> base
        [(Just declr@(CDeclr Nothing _ _ _ _), Nothing, Nothing)] ->
            derivedTypeOf base declr
        _ -> badSource decl "type name"
    when (typeIsFunc itype) (badSource decl "use of function type")
    return (typeMutable itype, typeRep itype)
```
