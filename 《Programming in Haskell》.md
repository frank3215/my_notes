本文件简单记录了《Programming In Haskell》中涵盖的内容，方便回顾。

## 2 First steps

### 2.1 Glasgow Haskell Compiler

-   GHC、GHCi

### 2.2 Installing and starting

-   [https://www.haskell.org/ghc/](https://www.haskell.org/ghc/)

### 2.3 Standard prelude

-   List

-   `head`（`last`）
-   `tail`（`init`）
-   `!!`
-   `drop`
-   `length`
-   `sum`
-   `product`
-   `++`
-   `reverse`

### 2.4 Function application

-   Haskell里，Function Application优先级最高，且不用显式写出

### 2.5 Haskell scripts

-   .hs后缀

**My first script**

-   建议一个窗口打开`test.hs`，一个窗口运行GHCi
-   `:reload`
-   常见命令：（可以简写成首字母）

-   `:load name`
-   `:reload`
-   `:set editor name`
-   `:edit name`
-   `:edit`
-   `:type expr`
-   `:?`
-   `:quit`

**Naming requirements**

-   函数与变量名：`[a-z][a-zA-Z_']*`

-   keywords
-   惯例：列表在名字后加`s`

**The layout rule**

-   可以用indentation指示level，也可以用`;`与`{}`指示。

**Tabs**

-   建议不要用tab——如果用的话，假设宽度是8

**Comments**

-   `--`
-   `{-`与`-}`

### 2.6 Chapter remarks

> In addition to the GHC system, [http://www.haskell.org](https://www.haskell.org/) contains a wide range of other useful resources concerning Haskell, including community activities, language documentation, and news items.

### 2.7 Exercises

## 3 Types and classes

### 3.1 Basic concepts

-   A _type_ is a collection of related values.

-   notation: `v :: T`——值`v`的类型是`T`
-   notation: `e :: T`——表达式`e`的结果的类型是`T`

-   _type inference_（类型推断）、_type error_（类型错误）

-   Haskell是_type safe_的——每个expression（表达式）的type（类型）在表达式求值之前计算。在**evaluation（求值）**时，永远不会发生_type error_
-   _type error_能检测到程序中的很多错误。
-   conditional expression（条件语句）的typing rule（类型规则）：两种分支返回的结果的type（类型）相同

### 3.2 Basic types

`Bool` – logical values

-   `False`与`True`

`Char` – single characters

-   包含Unicode中所有字符。
-   语法：单引号`' '`。

`String` – strings of characters

-   语法：双引号`" "`。

`Int` – fixed-precision integers

-   范围：[-2^{63},2^{63}-1][−2^{63},2^{63}−1]——求值`2^63 :: Int`时会报错

-   `:: T`强制使表达式的值为类型`T`

`Integer` – arbitrary-precision integers

-   没有上下界或内存上限。
-   `Int`运算（常硬件）一般比`Integer`运算（常软件）快

`Float` – single-precision floating-point numbers

-   浮点数：需要的内存固定。The term floating-point comes from the fact that the number of digits permitted after the decimal point depends upon the size of the number.

`Double` – double-precision floating-point numbers

-   双精度：需要的内存固定。

### 3.3 List types

-   列表：一列类型相同的元素。

-   notation:`[T]`
-   empty list, singleton list
-   长度可以无限长

### 3.4 Tuple types

-   元组：一列长度有限且固定的元素，其类型可以不同。

-   arity：元组的元素数量
-   不允许出现长度为1的元组

-   三点：

1.  元组的类型蕴含了它的arity
2.  可以有元组的元组，也可以有列表的元组、元组的列表
3.  元组的arity必定有限——因为需要在求值之前计算表达式的类型

### 3.5 Function types

-   `T1 -> T2`：“将`T1`类型的参数映射到`T2`类型的值的函数”的类型

-   惯例：把函数类型放在函数定义之前——有用的documentation（文档）

-   注意：函数不必对所有可能的参数都返回结果。

### 3.6 Curried functions

-   定义：Functions such as add’ and mult that take their arguments one at a time are called curried functions.
-   惯例：

-   `->`右结合
-   函数应用左结合

-   除非明确写出Tupling，Haskell的函数一般都被定义为curried function

### 3.7 Polymorphic types

-   _type variable_（类型变量）
-   polymorphic（多态的/多模态的）：A type that contains one or more type variables is called _polymorphic_(“of many forms”), as is an expression with such a type.

### 3.8 Overloaded types

-   _class constraint_：Class constraints are written in the form `C a`, where `C` is the name of a class and `a` is a type variable.
-   _overloaded type_A type that contains one or more class constraints is called _overloaded_, as is an expression with such a type.

### 3.9 Basic classes

-   `Eq` – equality typestypes whose values can be compared for equality

-   `==`、`/=`

-   `Ord` – ordered typesalso of `Eq` type, totally ordered, including `List`s and `Tuple`s

-   `(<) (<=) (>) (>=) (min) (max)`

-   `Show` – showable types

-   `show`函数

-   `Read` – readable types

-   `read`函数

-   `Num` – numeric types

-   `(+) (-) (*) negate abs signum`

-   `Integral` – integral types

-   `div mod`
-   包括（但不止）`Int`

-   `Fractional` – fractional types

-   `(/) recip`

### 3.10 Chapter remarks

-   GeorgeBoole：符号逻辑
-   HaskellCurry：curry化（柯里化）

### 3.11 Exercises

-   1.`[tail, init, reverse]`的type？
-   2.`apply :: (a -> b) -> a -> b`的一种可能的函数定义？
-   3.`twice f x = f (f x)`的type？

## 4 Defining functions

### 4.1 New from old

> Perhaps the most straightforward way to define new functions is simply by combining one or more existing functions. For example, a few library functions that can be defined in this way are shown below.

### 4.2 Conditional expressions

```haskell
signum :: Int -> Int
signum n = if n < 0 then -1 else
                 if n == 0 then 0 else 1
```

> conditional expressions in Haskell must **always have an else branch**, which avoids the well-known dangling else problem.

### 4.3 Guarded equations

> The symbol `|` is read as _such that_, and the guard otherwise is defined in the standard prelude simply by `otherwise = True`  
> The main benefit of guarded equations over conditional expressions is that definitions with multiple guards are easier to read.

### 4.4 Pattern matching

-   one name can not be used as two arguments: e.g.`f x x = x + x` is invalid

-   the definition could be made valid by using guard equations

Tuple patterns

```haskell
fst :: (a,b) -> a
fst (x,_) = x
```

List patterns

对3.3 List types的注脚：

-   a list is constructed from an empty list, one element of a time:

```haskell
[1,2,3] == 1:2:3:[]
head :: [a] -> a
head (x:_) = x
tail :: [a] -> [a]
tail (_:xs) = xs
```

> Note that cons patterns must be parenthesised, because function application has higher priority than all other operators in the language.

### 4.5 Lambda expressions

-   LambdaExpression

```haskell
\x -> x + x
```

Applications

-   formalising curried function definitions(?)
-   defining functions that return functions by their nature(?)
-   avoid having to name functions (such as when using `map`)

### 4.6 Operator sections

-   operator
-   operator section

Applications:

1.  contructing useful functions in a compact way
2.  stating the type of operators
3.  being used as arguments to other functions

### 4.7 Chapter remarks

> A formal meaning for pattern matching by translation using more primitive features of the language is given in the Haskell Report [4]. The Greek letter λ used when defining nameless functions comes from the lambda calculus [6], the mathematical theory of functions upon which Haskell is founded.

[4] S. Marlow, Ed., Haskell Language Report, 2010, available on the web from: [https://www.haskell.org/definition/haskell2010.pdf](https://www.haskell.org/definition/haskell2010.pdf).

[6] H. Barendregt, The Lambda Calculus, Its Syntax and Semantics. North Holland, 1985.

### 4.8 Exercises

## 5 List comprehensions

### 5.1 Basic concepts

-   comprehension
-   `x <- [1..5]`：generator
-   `|` such that
-   `<-`：drawn from

```haskell
> [(x,y) | y <- [4,5], x <- [1,2,3]]
[(1,4),(2,4),(3,4),(1,5),(2,5),(3,5)]
```

-   later generators are iterated first, and can depend on previous generators:`concat xss = [x | xs <- xss, x <- xs]`
-   pattern matching can be used:`firsts ps = [x | (x,_) <- ps]`

### 5.2 Guards

> List comprehensions can also use logical expressions called guards to filter the values produced by

earlier generators.

```haskell
factors :: Int -> [Int]
factors n = [x | x <- [1..n], n ‘mod‘ x == 0

prime :: Int -> Bool
prime n = factors n == [1,n]
```

-   note: under lazy evaluation, as soon as the second factor is found(and is not n), the function `prime` terminates.

```haskell
find :: Eq a => a -> [(a,b)] -> [b]
find k t = [v | (k’,v) <- t, k == k’]
```

### 5.3 The zip function

```haskell
pairs :: [a] -> [(a,a)]
pairs xs = zip xs (tail xs)

sorted :: Ord a => [a] -> Bool
sorted xs = and [x <= y | (x,y) <- pairs xs]

positions :: Eq a => a -> [a] -> [Int]
positions x xs = [i | (x',i) <- zip xs [0..], x == x']
```

-   note: the last example exploits lazy evaluation

### 5.4 String comprehensions

-   `String` is a shorthand for `[Char]`

e.g.

```haskell
lowers :: String -> Int
lowers xs = length [x | x <- xs, x >= 'a' && x <= 'z']

count :: Char -> String -> Int
count x xs = length [x' | x' <- xs, x == x']
```

### 5.5 The Caesar cipher

Encoding and decoding

```haskell
let2int :: Char -> Int
let2int c = ord c - ord 'a'


int2let :: Int -> Char
int2let n = chr (ord 'a' + n)

shift :: Int -> Char -> Char
shift n c | isLower c = int2let ((let2int c + n) `mod` 26)
          | otherwise = c

encode :: Int -> String -> String
encode n xs = [shift n x | x <- xs]
```

Frequency tables

```haskell
table :: [Float]
table = [8.1, 1.5, 2.8, 4.2, 12.7, 2.2, 2.0, 6.1, 7.0,
 0.2, 0.8, 4.0, 2.4, 6.7, 7.5, 1.9, 0.1, 6.0,
 6.3, 9.0, 2.8, 1.0, 2.4, 0.2, 2.0, 0.1]

percent :: Int -> Int -> Float
percent n m = (fromIntegral n / fromIntegral m) * 100

freqs :: String -> [Float]
freqs xs = [percent (count x xs) n | x <- ['a'..'z']]
 where n = lowers xs
```

`count`和`lowers`来自5.4 String comprehensions

Cracking the cipher

-   chi-square statistic

```haskell
chisqr :: [Float] -> [Float] -> Float
chisqr os es = sum [((o-e)^2)/e | (o,e) <- zip os es]

rotate :: Int -> [a] -> [a]
rotate n xs = drop n xs ++ take n xs

crack :: String -> String
crack xs = encode (-factor) xs
 where
  factor = head (positions (minimum chitab) chitab)
  chitab = [chisqr (rotate n table') table | n <- [0..25]]
  table' = freqs xs
```

## 6 Recursive functions

### 6.1 Basic concepts

-   _recursive_ functions:

-   _base case_
-   _recursive case_

-   fact函数
-   乘函数

### 6.2 Recursion on lists

```haskell
product :: Num a => [a] -> a
product [] = 1
product (n:ns) = n * product ns
length :: [a] -> Int
length [] = 0
length (_:xs) = 1 + length xs
reverse :: [a] -> [a]
reverse [] = []
reverse (x:xs) = reverse xs ++ [x]
(++) :: [a] -> [a] -> [a]
[] ++ ys = ys
(x:xs) ++ ys = x : (xs ++ ys)
insert :: Ord a => a -> [a] -> [a]
insert x []
insert x (y:ys) | x <= y    = x : y : ys
                | otherwise = y : insert x ys
isort :: Ord a => [a] -> [a]
isort [] = []
isort (x:xs) = insert x (isort xs)
```

### 6.3 Multiple arguments

```haskell
zip :: [a] -> [b] -> [(a,b)]
zip []     _      = []
zip _      []     = []
zip (x:xs) (y:ys) = (x,y) : zip xs ys
drop :: Int -> [a] -> [a]
drop 0 xs     = xs
drop _ []     = []
drop n (_:xs) = drop (n-1) xs
```

### 6.4 Multiple recursion

```haskell
fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n-2) + fib (n-1)
qsort :: Ord a => [a] -> [a]
qsort []     = []
qsort (x:xs) = qsort smaller ++ [x] ++ qsort larger
               where
                  smaller = [a | a <- xs, a <= x]
                  larger  = [b | b <- xs, b > x]
```

### 6.5 Mutual recursion

```haskell
even :: Int -> Bool
even 0 = True
even n = odd (n-1)

odd :: Int -> Bool
odd 0 = False
odd n = even (n-1)
evens :: [a] -> [a]
evens []     = []
evens (x:xs) = x : odds xs

odds :: [a] -> [a]
odds []     = []
odds (x:xs) = evens xs
```

### 6.6 Advice on recursion

Steps:

1.  define the type
2.  enumerate the cases
3.  define the simple cases
4.  define the other cases
5.  generalise and simplify

```haskell
product :: Num a => [a] -> a
product = foldr (*) 1
drop :: Int -> [a] -> [a] -- 因为效率原因不用Integral
drop 0 xs     = xs
drop _ []     = []
drop n (_:xs) = drop (n-1) xs
init :: [a] -> [a]
init [_] = []
init (x:xs) = x : init xs
```

## 7 Higher-order functions

### 7.1 Basic concepts

-   some examples given by the book exploits the fact that functions are curried.

> a function that takes a function as an argument or returns a function as a result is called a _higher-order function_. In practice, however, because the term curried already exists for returning functions as results, the term higher-order is often just used for taking functions as arguments. It is this latter interpretation that is the subject of this chapter.

-   higher-order functions can be used to define domain-specific languages within Haskell.

### 7.2 Processing lists

```haskell
map :: (a -> b) -> [a] -> [b]
map f xs = [f x | x <- xs]
```

1.  it is a polymorphic function that can be applied to lists of any type
2.  it can be applied to itself to process nested lists
3.  the function map can also be defined using recursion

```haskell
filter :: (a -> Bool) -> [a] -> [a]
filter p xs = [x | x <- xs, p x]
```

### 7.3 The foldr function

```haskell
foldr :: (a -> b -> b) -> b -> [a] -> b
foldr f v [] = v
foldr f v (x:xs) = f x (foldr f v xs)
```

-   fold right

```haskell
foldr (#) v [x0,x1,...,xn]=x0 # (x1 # (... (xn # v) ...))
```

### 7.4 The foldl function

```haskell
foldl :: (a -> b -> a) -> a -> [b] -> a
foldl f v [] = v
foldl f v (x:xs) = foldl f (f v x) xs
```

> When a function can be defined using both foldr and foldl, as in the above examples, the choice of which definition is preferable is usually made on grounds of efficiency and requires careful consideration of the evaluation mechanism underlying Haskell, which is discussed in chapter 15.

```haskell
foldl (#) v [x0,x1,...,xn]=(... ((v # x0) # x1) ...) # xn
```

### 7.5 The composition operator

```haskell
(.) :: (b -> c) -> (a -> b) -> (a -> c)
f . g = \x -> f (g x)
```

-   used to simplify nested function applications
-   is associative: `f . (g . h) = (f . g) . h`

identity function:

```haskell
id :: a -> a
id = \x -> x
```

usage:

```haskell
compose :: [a -> a] -> (a -> a)
compose = foldr (.) id
```

### 7.6 Binary string transmitter

Binary numbers

-   首位是最低位（权是1）

Base conversion

```haskell
import Data.Char

type Bit = Int

bin2int :: [Bit] -> Int
bin2int = foldr (\x y -> x + 2*y) 0

int2bin :: Int -> [Bit]
int2bin 0 = []
int2bin n = n `mod` 2 : int2bin (n `div` 2)

make8 :: [Bit] -> [Bit]
make8 bits = take 8 (bits ++ repeat 0)
```

Transmission

```haskell
encode :: String -> [Bit]
encode = concat . map (make8 . int2bin . ord)

chop8 :: [Bit] -> [[Bit]]
chop8 [] = []
chop8 bits = take 8 bits : chop8 (drop 8 bits)

decode :: [Bit] -> String
decode = map (chr . bin2int) . chop8

transmit :: String -> String
transmit = decode . channel . encode

channel :: [Bit] -> [Bit]
channel = id
```

### 7.7 Voting algorithms

First past the post

```haskell
votes :: [String]
votes = ["Red", "Blue", "Green", "Blue", "Blue", "Red"]

count :: Eq a => a -> [a] -> Int
count x = length . filter (== x)

rmdups :: Eq a => [a] -> [a]
rmdups [] = []
rmdups (x:xs) = x : filter (/= x) (rmdups xs)

result :: Ord a => [a] -> [(Int,a)]
result vs = sort [(count v vs, v) | v <- rmdups vs]

winner :: Ord a => [a] -> a
winner = snd . last . result
```

Alternative vote

```haskell
rmempty :: Eq a => [[a]] -> [[a]]
rmempty = filter (/= [])

elim :: Eq a => a -> [[a]] -> [[a]]
elim x = map (filter (/= x))

rank :: Ord a => [[a]] -> [a]
rank = map snd . result . map head

winner' :: Ord a => [[a]] -> a
winner' bs = case rank (rmempty bs) of
    [c] -> c
    (c:cs) -> winner' (elim c bs)
```

case: pattern matching in the body of a definition

## 8 Declaring types and classes

### 8.1 Type declarations

```haskell
type Pos = (Int,Int)
type Trans = Pos -> Pos
```

注意：`type`不能有递归——以下定义是不被允许的（但用`data`是可以的）：

```haskell
type Tree = (Int,[Tree])
```

`type`定义中可以有其他type的参数：

```haskell
type Assoc k v = [(k,v)]
find :: Eq k => k -> Assoc k v -> v
find k t = head [v | (k’,v) <- t, k == k’]
```

注意：和C++中的`typedef`类似，只是个假名而已。

### 8.2 Data declarations

```haskell
data Bool = False | True
```

-   `|`：“or”
-   只要这些名字没有用过就行

自定义的类型可以随便用——和Haskell自带的类型一样。

-   `deriving Show`

一个`data`的定义可以特别丑陋……

```haskell
data Shape = Circle Float | Rect Float Float

square :: Float -> Shape
square n = Rect n n

area :: Shape -> Float
area (Circle r) = pi * r^2
area (Rect x y) = x * y
```

-   `Circle`和`Rect`其实是_constructor functions_

-   constructer function没有定义式，和普通function不同

```haskell
> :type Circle
Circle :: Float -> Shape
> :type Rect
Rect :: Float -> Float -> Shape
data Maybe a = Nothing | Just a

safediv :: Int -> Int -> Maybe Int
safediv _ 0 = Nothing
safediv m n = Just (m ‘div‘ n)

safehead :: [a] -> Maybe a 
safehead [] = Nothing
safehead xs = Just (head xs)
```

### 8.3 Newtype declarations

```haskell
newtype Nat = N Int
```

-   一个参数
-   与`type`的区别：新type，而非假名
-   与`data`的区别：效率较高（`N`在检查完type后被移除）

### 8.4 Recursive types

定义：

1.  可以有多个case——可以把若干个type放在一起，搞成一个大杂烩type。

```haskell
data SomeType typeVariable = SomeValue | MyConstructor Float | SomeConstructor typeVariable (SomeType typeVariable)
```

定义函数：用的是pattern maching，就按Constructor定义就好了。

-   例子：

-   `data Nat`
-   `data List a`
-   `data Tree a`

-   `occurs`
-   `flatten`

-   _search tree_

-   `data Tree a = Node a [Tree a]`

### 8.5 Class and instance declarations

`class`

-   `class Eq a where...`

-   _default definition_

-   可以被overridden

-   `instance Eq Bool where...`

-   仅限`data`和`newtype`

-   `class Eq a => Ord a where...`

Derived instances

-   `data Bool = False | True deriving (Eq, Ord, Show, Read)`

-   `Ord`和元素顺序有关（字典序）

### 8.6 Tautology checker

-   `data Prop`
-   `eval`

-   `Subst`,`Assoc`

-   `vars`

-   `rmdump`

-   `bools`
-   `substs`
-   `isTaut`

### 8.7 Abstract machine

-   `data Expr`
-   `value :: Expr -> Int`
-   _control stack_`type Cont = [Op]`
-   `eval`
-   `exec`

……没怎么理解……

## 9 The countdown problem

## 10 Interactive programming

## 11 Unbeatable tic-tac-toe

## 12 Monads and more

### 12.1 Functors

-   `inc`,`sqr`

-   `map`

> The class of types that support such a mapping function are called _functors_

```haskell
class Functor f where
  fmap :: (a -> b) -> f a -> f b
instance Functor [] where
  -- fmap :: (a -> b) -> [a] -> [b]
  fmap = map
```

-   _container type_
-   `IO` as a functor???

Functor laws

-   preserve identity: `fmap id = id`
-   preserve function composition:`fmap (g . h) = fmap g . fmap h`

真的吗？？：

> In fact, for any parameterised type in Haskell, there is at most one function fmap that satisfies the required laws.

### 12.2 Applicatives

-   `fmap0`,`fmap1`,`fmap2`,...
-   _applicative style_
-   _applicative functors_(_applicatives_)The class of functors that support pure and <*> functions are called applicative functors, or applicatives for short.
-   `pure`,`<*>`

Examples

-   _non-deterministic_ programming
-   _interactive_ programming

Effectful programming

-   programming with _effects_the arguments are no longer just plain values but may also have effects, such as the possibility of failure, having many ways to succeed, or performing input/output actions.  
    
-   ???他在说啥……完蛋了我不懂了……

-   `sequenceA`

-   ??? !!!

Applicative laws

```haskell
pure id <*> x   = x -- preserve identity equation 
pure (g x)      = pure g <*> pure x -- 
x <*> pure y    = pure (\g -> g y) <*> x
x <*> (y <*> z) = (pure (.) <*> x <*> y) <*> z
```

-   ???The third equation states that when an effectful function is applied to a pure argument, the order in which we evaluate the two components doesn’t matter.
-   好妙！In particular, the fourth law reassociates applications to the left, the third law moves occurrences of pure to the left, and the remaining two laws allow zero or more consecutive occurrences of pure to be combined into one.
-   `fmap g x = g <$> x = pure g <*> x`

### 12.3 Monads

-   `Maybe`

-   applicative: not useful
-   `>>=`: bind

```haskell
m1 >>= \x1 ->
m2 >>= \x2 ->
.
.
.
mn >>= \xn ->
f x1 x2 ... xn
```

在Haskell里可以写作

```haskell
do x1 <- m1
  x2 <- m2
  .
  .
  .
  xn <- mn
  f x1 x2 ... xn
class Applicative m => Monad m where 
  return :: a -> m a
  (>>=) :: m a -> (a -> m b) -> m b

  return = pure
```

我有点看不懂下面那个为什么能支持`->`符号：

```haskell
instance Monad [] where
  -- (>>=) :: [a] -> (a -> [b]) -> [b]
  xs >>= f = [y | x <- xs, y <- f x]
```

当Monad是一个可以有副作用的东西的时候，我勉强能理解`>>=`与`return`……但为什么`[]`也是个Monad？？？？

在其他例子中，`x <- m`表示`m >>= \x ->`，但在`[]`的例子中，`<-`似乎表示`>>=`？？？

-   还是好难理解啊……

```haskell
getChar >>= \x ->
getChar >>= \y ->
return (x,y)

do
  x <- getChar
  y <- getChar
  return (x,y)

[1,2]
```

怎么把`Maybe`变成Monad？？？——哦它讲了

**Examples**

```haskell
instance Monad Maybe where
  -- (>>=) :: Maybe a -> (a -> Maybe b) -> Maybe b
  Nothing >>= _ = Nothing
  (Just x) >>= f = f x
```

**The state monad**

-   `ST`（和`IO`非常像）

-   有很多直观的图片……你早放啊！！！（我理解`IO`的时候快死了……）

-   `ST`: applicative functor

-   Applicative必定是一个Functor吗？？？
-   `$`: function application
-   ！！同学提出了一个有价值的想法：调换ST的apply在上面的顺序，但不调换type，也是一个applicative！（应该也是一个monad）

我开始不理解求值顺序了……Haskell的懒求值和这个有什么关系啊……

……好玄妙啊……虽然我感觉我能理解它的实际应用，但我感觉我并没有完全理解它……

等下……确实`\`与`>>=`并不能保证求值顺序！！

**Relabelling trees**

-   stateful programming

```haskell
fresh :: ST Int
fresh = S (\n -> (n, n+1))
alabel :: Tree a -> ST (Tree Int)
alabel (Leaf _) = Leaf <$> fresh
alabel (Node l r) = Node <$> alabel l <*> alabel r
```

……我感觉我并没有完全理解……先这样吧……

```haskell
mlabel :: Tree a -> ST (Tree Int)
mlabel (Leaf _) = do n <- fresh
  return (Leaf n)
mlabel (Node l r) = do
  l' <- mlabel l
  r' <- mlabel r
  return (Node l' r')
```

……看起来好像C++啊……能不能直接那么理解啊……

……如果我证明一遍它的正确性，我的理解会不会更深？——算了待会再说吧。（也可以直接上网查）

**Generic functions**

```haskell
mapM :: Monad m => (a -> m b) -> [a] -> m [b]
mapM f [] = return []
mapM f (x:xs) = do y <- f x
  ys <- mapM f xs
  return (y:ys)
filterM :: Monad m => (a -> m Bool) -> [a] -> m [a]
filterM p [] = return []
filterM p (x:xs) = do b <- p x
  ys <- filterM p xs
  return (if b then x:ys else ys)
filterM (\x -> [True,False]) -- [1,2,3] [[1,2,3],[1,2],[1,3],[1],[2,3],[2],[3],[]]
```

我都想象不出来`filterM`与`mapM`可以被用来干什么（没有直观感觉）……只能想象出对于一个特定实例它是什么意思……这是不是说明它特别强大啊……

```haskell
join :: Monad m => m (m a) -> m a
join mmx = do mx <- mmx
  x <- mx
  return x
```

**Monad laws**

-   `return x >>= f == f x`

-   ???……对`Maybe`来说靠谱，对`[]`靠谱，对`ST`……有点难以想象了……好像还是靠谱的……

-   `mx >>= return == mx`

-   能理解……

-   `(mx >>= f) >>= g == mx >>= (\x -> (f x >>= g))`

-   ??????……算了先看看它的解释吧……

> Together, these two equations state, modulo the fact that the second argument to >>= involves a binding operation, that return is the identity for the >>= operator.  
> The third equation concerns the link between >>= and itself, and expresses (again modulo binding) that >>= is associative.

## 13 Monadic parsing

### 13.1 What is a parser?

-   就是一个可以生成parse tree的玩意儿。

### 13.2 Parsers as functions

1.  String->Tree
2.  String->(Tree,String)
3.  String->[(Tree,String)]

-   `type Parser a = String -> [(a,String)]`
-   note: similar to`ST`:`State -> (a,State)`, but can fail or return morre than one result - generalized.

### 13.3 Basic definitions

```haskell
newtype Parser a = P (String -> [(a,String)])

parse :: Parser a -> String -> [(a,String)]
parse (P p) inp = p inp

item :: Parser Char
item = P (\inp -> case inp of
  [] -> []
  (x:xs) -> [(x,xs)])
```

### 13.4 Sequencing parsers

**Parser**

```haskell
instance Functor Parser where
  -- fmap :: (a -> b) -> Parser a -> Parser b
  fmap g p = P (\inp -> case parse p inp of
    [] -> []
    [(v,out)] -> [(g v, out)])
```

感性理解：如果parse挂了，那依然是“挂了”，否则把function apply到结果上。

**Applicative**

```haskell
instance Applicative Parser where
  -- pure :: a -> Parser a
  pure v = P (\inp -> [(v,inp)])

  -- <*> :: Parser (a -> b) -> Parser a -> Parser b
  pg <*> px = P (\inp -> case parse pg inp of
    [] -> []
    [(g,out)] -> parse (fmap g px) out)
```

感性理解（返回的parser的功能）：如果parse挂了，就返回“挂了”，否则继续parse，并把parse出来的function apply到下一个结果上。

例子：

```haskell
three :: Parser (Char,Char)
three = pure g <*> item <*> item <*> item
  where g x y z = (x,z)
```

有

```haskell
> parse three "ab"
[]
```

**Monad**

```haskell
instance Monad Parser where
  -- (>>=) :: Parser a -> (a -> Parser b) -> Parser 
  p >>= f = P (\inp -> case parse p inp of
    [] -> []
    [(v,out)] -> parse (f v) out)
```

理性理解失败了……

感性理解（返回的parser的功能）：如果第一个parser挂了，就返回“挂了”，否则，把余下的string和v继续传递下去。

例子：

```haskell
three :: Parser (Char,Char)
three = do x <- item
  item
  z <- item
  return (x,z)
```

### 13.5 Making choices

如果parser A挂了，就继续应用parser B——

**Alternative**

```haskell
class Applicative f => Alternative f where
  empty :: f a
  (<|>) :: f a -> f a -> f a
```

laws：

```haskell
empty <|> x     = x
x <|> empty     = x
x <|> (y <|> z) = (x <|> y) <|> z
```

例子：

```haskell
instance Alternative Maybe where
  -- empty :: Maybe a
  empty = Nothing

  -- (<|>) :: Maybe a -> Maybe a -> Maybe a
  Nothing <|> my = my
  (Just x) <|> _ = Just x
instance Alternative Parser where
  -- empty :: Parser a
  empty = P (\inp -> [])

  -- (<|>) :: Parser a -> Parser a -> Parser a
  p <|> q = P (\inp -> case parse p inp of
  [] -> parse q inp
  [(v,out)] -> [(v,out)])
```

### 13.6 Derived primitives

-   `item`, `return`, `empty`

```haskell
sat :: (Char -> Bool) -> Parser Char
sat p = do x <- item
  if p x then return x else empty
digit :: Parser Char
digit = sat isDigit

lower :: Parser Char
lower = sat isLower

upper :: Parser Char
upper = sat isUpper

letter :: Parser Char
letter = sat isAlpha

alphanum :: Parser Char
alphanum = sat isAlphaNum

char :: Char -> Parser Char
char x = sat (== x)
```

用`char`定义一个`string`：尝试从input里把一个string吃掉，如果吃不掉，就“挂了”。

```haskell
string :: String -> Parser String
string []     = return []
string (x:xs) = do char x
                   string xs
                   return (x:xs)
```

`many`和`some`已经被`Alternative`定义好了：

```haskell
class Applicative f => Alternative f where
  empty :: f a
  (<|>) :: f a -> f a -> f a
  many :: f a -> f [a]
  some :: f a -> f [a]

  many x = some x <|> pure []
  some x = pure (:) <*> x <*> many x
```

？？？这是什么诡异的定义方式。

……emm不过我记不住这种定义`class`的方式……Note to self: 待会回去复习一下。

……已经忘记`mapM`和`filterM`可以干什么了——它们分别对`[]`、`ST`与`Maybe`有什么意义？？——好像待会就会讲[捂脸]

```haskell
ident :: Parser String
ident = do x  <- lower
           xs <- many alphanum
           return (x:xs)

nat :: Parser Int
nat = do xs <- some digit
         return (read xs)

space :: Parser ()
space = do many (sat isSpace)
           return ()
```

nat返回Int，space返回()。

```haskell
int :: Parser Int
int = do char '-'
         n <- nat
         return (-n)
       <|> nat
```

### 13.7 Handling spacing

```haskell
token :: Parser a -> Parser a
token p = do space
             v <- p
             space
             return v
identifier :: Parser String
identifier = token ident

natural :: Parser Int
natural = token nat

integer :: Parser Int
integer = token int

symbol :: String -> Parser String
symbol xs = token (string xs)
```

例子：

```haskell
nats :: Parser [Int]
nats = do symbol "["
  n <- natural
  ns <- many (do symbol "," natural)   symbol "]"
  return (n:ns)
```

### 13.8 Arithmetic expressions

代码：`Code/expression.hs`，`import Parsing`。

```haskell
expr ::= expr+expr | expr*expr | (expr) | nat
nat ::= 0 | 1 | 2 | ...
```

——但依然存在多种parse tree，加入优先级：

```haskell
expr ::= expr+expr | term
term ::= term*term | factor
factor ::= (expr) | nat
nat ::= 0 | 1 | 2 | ...
```

加入右结合：

```haskell
expr ::= term+expr | term
term ::= factor*term | factor
```

简化：

```haskell
expr ::= term (+expr | )
term ::= factor (*term | )
factor ::= (expr) | nat
nat ::= 0 | 1 | 2 | ...
```

翻译：

```haskell
expr :: Parser Int
expr = do t <- term
          do symbol "+"
             e <- expr
             return (t + e)
           <|> return t

term :: Parser Int
term = do f <- factor
          do symbol "*"
             t <- term
             return (f * t)
           <|> return f

factor :: Parser Int
factor = do symbol "("
            e <- expr
            symbol ")"
            return e
          <|> natural
```

太简洁了！！就像定义了一种语言！

```haskell
eval :: String -> Int
eval xs = case (parse expr xs) of
             [(n,[])]  -> n
             [(_,out)] -> error ("Unused input " ++ out)
             []        -> error "Invalid input"
```

---

开头的：

```haskell
expr = expr + expr | 1
```

这样会出现无限循环吗？……好像还真会……

### 13.9 Calculator

代码：`Code/calculator.hs`

```haskell
import Parsing
import System.IO
```

用**10 Interactive programming**的`cls`，`writeat`，`goto`，`getCh`

```haskell
box :: [String]
box = ["+---------------+",
       "|               |",
       "+---+---+---+---+",
       "| q | c | d | = |",
       "+---+---+---+---+",
       "| 1 | 2 | 3 | + |",
       "+---+---+---+---+",
       "| 4 | 5 | 6 | - |",
       "+---+---+---+---+",
       "| 7 | 8 | 9 | * |",
       "+---+---+---+---+",
       "| 0 | ( | ) | / |",
       "+---+---+---+---+"]
buttons :: String
buttons = standard ++ extra
          where
             standard = "qcd=123+456-789*0()/"
             extra    = "QCD \ESC\BS\DEL\n"
showbox :: IO ()
showbox = sequence_ [writeat (1,y) b | (y,b) <- zip [1..] box]
```

……记得writeat用了一堆escape sequence来着……在哪里可以查到呢？

```haskell
display xs = do writeat (3,2) (replicate 13 ' ')
                writeat (3,2) (reverse (take 13 (reverse xs)))
calc :: String -> IO ()
calc xs = do display xs 
             c <- getCh
             if elem c buttons then
                 process c xs
             else
                 do beep
                    calc xs
```

其中`beep = putStr "\BEL"`。

```haskell
process :: Char -> String -> IO ()
process c xs | elem c "qQ\ESC"    = quit
             | elem c "dD\BS\DEL" = delete xs
             | elem c "=\n"       = eval xs
             | elem c "cC"        = clear
             | otherwise          = press c xs
calc :: String -> IO ()
calc xs = do display xs 
             c <- getCh
             if elem c buttons then
                 process c xs
             else
                 do beep
                    calc xs

process :: Char -> String -> IO ()
process c xs | elem c "qQ\ESC"    = quit
             | elem c "dD\BS\DEL" = delete xs
             | elem c "=\n"       = eval xs
             | elem c "cC"        = clear
             | otherwise          = press c xs
 
quit :: IO ()
quit = goto (1,14)

delete :: String -> IO ()
delete [] = calc []
delete xs = calc (init xs)

eval :: String -> IO ()
eval xs = case parse expr xs of
             [(n,[])] -> calc (show n)
             _        -> do beep
                            calc xs
 
clear :: IO ()
clear = calc []

press :: Char -> String -> IO ()
press c xs = calc (xs ++ [c])
```

顶层：

```haskell
run :: IO ()
run = do cls
         showbox
         clear
```

## 14 Foldables and friends

### 14.1 Monoids

-   幺半群。

-   满足：单位元、结合律

-   吐嘈了mappend和menpty的名字不好
-   注：可以顺便定义`Semigroup`和`<>`。

### 14.2 Foldables

-   Foldable

-   `fold`
-   `foldMap`
-   `foldr`
-   `foldl`

Examples

-   `Foldable []`
-   `Foldable Tree`

Other primitives and defaults

-   `null`,`length`,`elem`,`maximum`,`minimum`,`sum`,`product`吐嘈：感觉都像`foldMap`可以定义出来的玩意
-   `foldr1`,`foldl1`
-   `toList`

-   可以用`toList`定义以上所有东西，可以用`foldMap`定义`toList`，用`foldr`定义`foldMap`
-   只需定义`foldMap`或`foldr`即可。

### 14.3 Traversables

-   `traverse`
-   `Traversable`

-   `traverse :: Applicative f => (a -> f b) -> t a -> f (t b)`

-   ???

**Examples**

-   `Traversable []`

-   `[Maybe Int]`

-   `Traversable Tree`

-   `Tree (Maybe Int)`

**Other primitives and defaults**

-   `sequenceA`

-   `sequenceA :: Applicative f => t (f a) -> f (t a)`
-   `sequenceA = traverse id`
-   `traverse g = sequenceA . fmap g`

-   `mapM :: Monad m => (a -> m b) -> t a -> m (t b)`

-   `mapM = traverse`

-   `sequence :: Monad m => t (m a) -> m (t a)`

-   `sequence = sequenceA`

> when declaring a new type it is also useful to consider whether it can be made into a traversable type

## 15 Lazy evaluation

### 15.1 Introduction

-   evaluation order**in Haskell** any two different ways of evaluating the same expression will always produce the same final value, provided that they both terminate.  
    
-   （要求：没有副作用）
-   例子：命令式语言有副作用导致值与求值顺序相关。

### 15.2 Evaluation strategies

-   reducible expression (redex)
-   innermost evaluation

-   arguments are always fully evaluated before functions are applied.
-   arguments are passed by value

-   outermost evaluation

-   allows functions to be applied before their arguments are evaluated
-   arguments are passed by name

-   _strict_ functions

Lambda expressions

-   reduction within the body of a function is only permitted once the function has been applied.

-   functions are viewed as black boxes

### 15.3 Termination

-   `inf = 1 + inf`
-   `fst (0,inf)`

> if there exists any evaluation sequence that terminates for a given expression, then call-by-name evaluation will also terminate for this expression, and produce the same final result.

### 15.4 Number of reductions

-   `square (1+2)`

-   call-by-name evaluation may require more reduction steps than call-by-value evaluation

arguments are evaluated precisely once using call-by-value evaluation, but may be evaluated many times using call-by-name.

-   using pointers!

-   ……我就知道！……虽然我花了一点实验……

The use of call-by-name evaluation in conjunction with sharing is known as _lazy evaluation_.

### 15.5 Infinite structures

-   `ones = 1 : ones`

-   注：`ones = [1] ++ ones`也可以。

-   _potentially infinite_ list

### 15.6 Modular programming

-   separate _control_ from _data_

-   我又猜到了！

the data is only evaluated as much as required by the control

-   突然想到：`replicate n x = x : replicate (n-1) x`和`replicate n x = (replicate (n-1) x) ++ [x]`的计算量虽然一样，但用起来计算量不同！

-   `primes = sieve [2..]`
-   ``sieve (p:xs) = p : sieve [x | x <- xs, x `mod` p /= 0]``

### 15.7 Strict application

-   `$!` the top-level of evaluation of the argument expression x is forced before the function f is applied.if the argument has a basic type, such as Int or Bool, then top-level evaluation is simply complete evaluation.  
    for a pair type such as (Int,Bool), evaluation is performed until a pair of expressions is obtained

> strict application is mainly used to improve the space performance of programs

-   sumwith v (x:xs) = (sumwith $! (v+x)) xs
-   `foldl'`(`Data.Foldable`)