# Crash Course on Natural Language Processing

## What is NLP?

* Branch of artificial intelligence concerned with analysis, understanding, and/or generation of human language
* Fuzzy boundary with machine learning, information retrieval, linguistics, and even cognitive science
* Applications in speech recognition, machine translation, text mining, web search, handwriting recognition

## Paradigms of NLP

* Rule-based  - let a linguist tell you
* Statistical - empirical, statistical models
* Neural      - deep neural networks

## Probability Primer
```
p(x)                          probability of x occurring
p(x,y)                        joint probability of x and y occurring
p(x|y)                        probability of x given y

p(x|y) = p(x,y) / p(y)        definition of conditional probability
p(x,y) = p(x|y) p(y)          chain rule
p(x|y) = p(y|x) p(x) / p(y)   Bayes' rule
```

## Language Model

w                              sequence of words w_1, w_2,...,w_n
p(w)                           a language model!

## Why would a language model be useful?
* Speech recognition ("recognize speech" vs "wreck a nice beach")

  `argmax_w p(w|x) = p(w|x) p(w) / p(x), where x is speech signal`

* Other examples?

## Markov Chains

* Markov property - future states of a process only depend on present state
* Markov chain - a sequence of events that satisfy the Markov property

## Weather Example

* Seattle
     | rain | sun
rain | .5   | .5
sun  | .9   | .1

* Los Angeles
     | rain | sun
rain | .3   | .7
sun  | .1   | .9

* What would Provo's transition matrix look like?

## Language Model using Markov Chains
order     number of terms in the n-gram
m         size of current state, m = order - 1)
```
p(w)  = p(w_1, w_2,...,w_n)
      = p(w_n, w_{n-1},...,w_1)
      = p(w_n | w_{n-1},...,w_1) p(w_{n-1},...,w_1)
      = p(w_n | w_{n-1},...,w_1) p(w_{n-1} | w_{n-2},...,w_1) p(w_{n-2},...,w_1)
      = \prod_{i=1}^n p(w_i | w_{i-1},...,w_1)
      = \prod_{i=1}^n p(w_i | w_{i-1},...,w_{i-m})
```
* HUGE simplifying assumption made on the last line
  * Correct? No!
  * Useful? Yes!

## N-Gram Models
order 1 (unigram) model
`p(rrssrs) = p(r) p(r) p(s) p(s) p(r) p(s)`

order 2 (bigram) model
`p(^rrssrs$) = p(r|^) p(r|r) p(s|r) p(s|s) p(r|s) p(s|r) p($|s)`

order 3 (trigram) model
`p(^^rrssrs$) = p(r|^^) p(r|^r) p(s|rr) p(s|rs) p(r|ss) p(s|sr) p($|rs)`

## Estimating a Markov Model

Suppose your data is:
```
^rssrr$
^rrrsss$
^rsr$
```
       | rain | sun | $   || total
rain   | 3    | 3   | 2   || 8
sun    | 2    | 3   | 1   || 6
^      | 3    | 0   | 0   || 3
```
p(^rs$) = p(r|^) p(s|r) p($|s)
        = 3/3    3/8    1/6
        = .0625
```
