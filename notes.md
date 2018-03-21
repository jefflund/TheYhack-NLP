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
```
w                              sequence of words w_1, w_2,...,w_n
p(w)                           a language model!
```

## Why would a language model be useful?
* Speech recognition ("recognize speech" vs "wreck a nice beach")

  `argmax_w p(w|x) = p(w|x) p(w) / p(x), where x is speech signal`

* Other examples?

## Markov Chains

* Markov property - future states of a process only depend on present state
* Markov chain - a sequence of events that satisfy the Markov property

## Weather Example

* Seattle

|      | rain | sun |
| ---- | ---- | --- |
| rain | .5   | .5  |
| sun  | .9   | .1  |

* Los Angeles

|      | rain | sun |
| ---- | ---- | --- |
| rain | .3   | .7  |
| sun  | .1   | .9  |

* Provo?

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
* Order 1 (unigram) model
`p(rrssrs) = p(r) p(r) p(s) p(s) p(r) p(s)`
* Order 2 (bigram) model
`p(^rrssrs$) = p(r|^) p(r|r) p(s|r) p(s|s) p(r|s) p(s|r) p($|s)`
* order 3 (trigram) model
`p(^^rrssrs$) = p(r|^^) p(r|^r) p(s|rr) p(s|rs) p(r|ss) p(s|sr) p($|rs)`


Note: `^` and `$` are special start/end symbols respectively

[code hack0]

## Estimating a Markov Model

Given training data `D = <w^1, w^2,...>`, how do you estimate `p(w)`?

Suppose your data is:
```
D = <
      ^rssrr$
      ^rrrsss$
      ^rsr$
    >
```

Count of each transition

|      | rain | sun | $ |
| ---- | ---- | --- | - |
| rain | 3    | 3   | 2 |
| sun  | 2    | 3   | 1 |
| ^    | 3    | 0   | 0 |

Count of each context

|      | total |
| ---- | ----- |
| rain | 8     |
| sun  | 6     |
| ^    | 3     |

```
p(^rs$) = p(r|^) p(s|r) p($|s)
        = 3/3    3/8    1/6
        = .0625
```

[code hack1]

## Language Model Evaluation

Given a model `p(w)` and test data `D` how do you validate?

* Evaluate likelihood of the data:
```
  p(D) = \prod_{w \in D} p(w)`
```
* Computation is unsable, so use log-likelihood instead:
``` 
  log p(D) = \sum_{w \in D} log p(w)
           = \sum_{w \in D} \sum_{w_i \in w} log p(w_i|w_{i-1},...,w_{i-m})
```
* Use log-likelihood to choose various parameters (e.g. order)

[code hack2]

## Proceedural Name Generator

* We can sample from `p(w)`
* Model doesn't care what the data is:
  * Words -> Characters
  * Sequences -> Names

[code hack3]

## Cool Results

* This one was harder than the others...sorry

* Now that it works, try this!
  * Play with the order. What happens?
  * Try it with other datasets.

## Final Product

[namegen]
