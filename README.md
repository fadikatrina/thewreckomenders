# The Wreckomenders - RecSys Group Project at DKE UM
Recommend, explain, evaluate for individuals and groups food recipes.

Slides:
Paper:

## Getting started
Using Python 3.7, package versions specified in requirements txt    
1. Create virtual env
2. Install requirements from mac/windows requirements
3. Download the dataset from https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions and place the extracted zip in `/Data`
4. Run `data_prep.py` (only once ever)
5. Run `main.py` (check the file for more details)

## Conventions
### Comments
- Todo s specified with `todo:importance`
- Research ideas specified with `exp:difficulty` (add a rule to recognise it in your idea)
- Enhancement suggestions specified with `enh:difficulty` (add a rule to recognise it in your idea)
### Logging
- Output to the console only `logger.debug`
- Output to the console during runtime AND saved to `logs/datetime.log` is logged with `logger.info` `logger.critical` `logger.error`
- Info only needed during runtime using `debug`
- Info for analysing the logs `info`
- Recommendations & explanations using `critical`
- Runtime warnings user needs to pay attention to and important comments for analysis `error`
### Directory Structure
- `rec_<type>_<name>`
- `expl_<type>_<name>`
- `eval_<type>_<name>`
- `script_<short desc>`
- `helpers_<short desc>`
- `filter_<short desc>`

## Experiments/research
- [ ]  Compare kNN vs SVD vs Content Based
- [ ]  Compare kNN Group recommender vs. UserUser CF group recommender vs. ItemItem CF group recommender
- [ ]  Compare group recommender aggregation strategies (probably qualitatively/inspection rather than quantitatively)
- [ ]  Group: individual member satisfaction compared to the others

## User Flow



