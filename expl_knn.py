import time

from logger import l


def indiv_CB(item_name, rating):
    l.critical(
        'The recipe "' + item_name + '" has been recommended to you since we think that this recipe might suit your '
                                     'taste')
    time.sleep(0.1)
    if int(input("Input 1 to get more details or enter to move on ") or "0"):
        l.critical('Our recommendation algorithm considered the nr of steps and cooking time of the recipes you have '
                   'rated before and figured that you would rate the aforementioned recipe with a rating of ' + rating)
