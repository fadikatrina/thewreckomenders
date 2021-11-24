# The Wreckomenders - RecSys Group Project at DKE UM
Recommend, explain, evaluate for individuals and groups food recipes.

## Getting started
Using Python 3.7, package versions specified in requirements txt    
1. Create virtual env
2. Install requirements from mac/windows requirements
3. Download the dataset from https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions and place the extracted zip in `/Data`
4. Run `data_prep.py` (once only)
5. Run `main.py` (see the file main function for more details)

## Conventions
### Comments
- Todo s specified with `todo:priority`
- Research ideas specified with `exp:difficulty` (add a rule to recognise it in your idea)
- Enhancement suggestions specified with `enh:difficulty` (add a rule to recognise it in your idea)
### Logging
- Output to the console during runtime AND saved to `logs/datetime.log` is logged with `logger.info` `logger.critical` `logger.error`
- Output to the console only `logger.debug`

## Every algorithm should have
- [ ]  Motivate the choice of the algorithm
- [ ]  Evaluate the algorithm qualitatively and against some sort of baseline or offline evaluation (what should the baseline be?)
- [ ]  Explanations to the user

## Experiments/research
- [ ]  Compare kNN vs SVD vs Content Based
- [ ]  Compare kNN Group recommender vs. UserUser CF group recommender vs. ItemItem CF group recommender
- [ ]  Compare group recommender aggregation strategies (probably qualitatively/inspection rather than quantitatively)

## Agenda Next Meeting/Think more about
- [ ]  Explanations
- [ ]  Evaluations (particularly, think of scenario for your group recommendation)
- [ ]  You could capture other aspects related to this (for instance, percentage of fat, calories, etc.), and show the trade-off between accuracy and health recommendations?

## User Flow
1. Select User/Group recommender
2. Filter minimum number of reviews the recipe and user should have
3. Filter the duration/number of steps
4. Filter healthy/not (minimum threshold)
5. Input user id(s), or select randomly
6. User has less than 5 reviews? ⇒ popularity based recommendations + explanation, ask for keywords, provided? ⇒ use content based approach, explanations, END
7. Group recommendation? ⇒ what is the type of relationship?
    1. Couples/Family/Very close friends ⇒ most pleasure
    2. Casual friends/partners ;) ⇒ average
    3. Acquaintances ⇒ least misery
8. Otherwise, choose algorithm
9. Algorithm output + explanations
10. Want to run another algorithm? ⇒ step 7
11. END

## Midterm Feedback

10, 5, 7, 8 = 7.5

First, we particularly appreciated the choice of the dataset and the deepen analysis you performed on aspects like sparsity and balance. Nice that you plan to use implicit feedback to tackle the sparsity problem. The proposed hybrid approach, using popularity based explanations to face the cold-start problem and switch to a content based when you have enough feedback, is interesting. You might have more information on how to implement that from the lecture on Hybrid recommenders next Thursday. Also, the idea of a post-filtering based on a healthy factor is really nice.
Regarding the group recommender, we suggest you think of a scenario in which your system is providing the recommendation. Regarding the explanations, this is the more critical point. We understand that spending so much time analysing the dataset and the recommendation strategies, this aspect was omitted till now. This is not a big problem, but we want to encourage you to start reasoning about it.

The evolution was partially motivated, although the metrics are still not decided. Here, you could reason about your research questions and which aspects you might want to evaluate: for instance, we can imagine that the healthy post-filtering will decrease the performances in terms of accuracy. But, you could capture other aspects related to this (for instance, percentage of fat, calories, etc.), and show the trade-off between accuracy and health recommendations? You can show this comparing your approach with appropriate baselines.
