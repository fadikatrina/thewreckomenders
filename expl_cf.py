import time

from logger import l


def expl_least_misery(item_name, unhealthy, rating):
    l.critical(
        'The recipe "' + item_name + '" has been recommended since we think that all of you will enjoy this recipe, '
                                     'it is healthy and none of the group members have a problem with it.')
    time.sleep(0.1)
    if int(input("Input 1 to get more details or enter to move on ") or "0"):
        l.critical('The aforementioned recipe is the highest rated healthy recipe, considering the minimum rating '
                   'among group members, namely ' + rating + '. The initial recommendation with the highest minimal '
                                                             'rating was "' + unhealthy +
                   '" but since its unhealthy we dont recommend it.')


def expl_approval_voting(item_name, votes, average_rating):
    l.critical('The recipe "' + item_name +
               '" has been recommended since it is approved by the highest amount of group members,'
               ' and has the highest rating among the recipes with highest approval rate.')
    time.sleep(0.1)
    if int(input("Input 1 to get more details or enter to move on ") or "0"):
        l.critical('All of the group members voted for the recipes that scored a predicted rating greater than 3.5. '
                   'The aforementioned recipe got an average score of ' + average_rating + ' and ' + votes +
                   ' approval votes')


def expl_most_pleasure(item_name, unhealthy, rating):
    l.critical('The recipe "' + item_name + '" has been recommended since some of you will enjoy this recipe, '
                                            'because it achieves the highest individual rating.')
    time.sleep(0.1)
    if int(input("Input 1 to get more details or enter to move on ") or "0"):
        l.critical('The aforementioned recipe is the highest rated healthy recipe, considering the maximum rating '
                   'among group members, namely ' + rating + '. The initial recommendation with the highest maximal '
                                                             'rating was "' + unhealthy +
                   '" but since its unhealthy we dont recommend it.')


def individual_user(item_name, rating):
    l.critical(
        'The recipe "' + item_name + '" because we think you might like it')
    time.sleep(0.1)
    if int(input("Input 1 to get more details or enter to move on ") or "0"):
        l.critical('The algorithm computed the similarity between you and other users, found the most similar K '
                   'users based on previous scores and predicted that the aforementioned item will be rated with'
                   'a rating of ' + rating + ' by you.')


def individual_item(item_name, rating):
    l.critical(
        'The recipe "' + item_name + '" because we think you might like it')
    time.sleep(0.1)
    if int(input("Input 1 to get more details or enter to move on ") or "0"):
        l.critical('The algorithm computed the similarity between the items you previously rated with other items, '
                   'found the most similar K '
                   'item based on previous scores and predicted that the aforementioned item will be rated with'
                   'a rating of ' + rating + ' by you.')
