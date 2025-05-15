"""Adaptation fo slinding window using ALS."""

# Standard library imports
import itertools  # noqa: I001
import operator

# Third-party library imports
import numpy as np

import pandas as pd

# Local application/library specific imports
from pygrex.config import cfg
from pygrex.data_reader import DataReader, GroupInteractionHandler
from pygrex.models.als_model import ALS

# Third-party library imports
from scipy import stats


class SlidingWindow:
    """Class for initiating and keeping track of the sliding window."""

    def __init__(self, arr, window_size):
        """Initiate the class.

        Args:
            arr (list): The list of items
            window_size (int): the size of the window. The size remains static
            throughout the experiment

        """
        self.arr = arr
        self.window_size = window_size
        self.index = 0  # Keep track of the current window

    # Returns the next window and set the index to point to the following window
    def get_next_window(self):
        if self.index + self.window_size <= len(self.arr):
            window = self.arr[self.index : self.index + self.window_size]
            self.index += 1  # Move to the next window
            return window
        else:
            return None  # Return None when all windows are processed


# Changes the original data by removing the items in the potential counterfactual explanation from all the interactions
# of the group members
# Input: originalData: the original data without any changes
#        groupIds: a list with the ids of the group members
#        itemIds: a list of items - the potential counterfactual explanation
# Output: newData: the altered data
def changeData(originalData, groupIds, itemIds, data=None):
    # check if originalData is a DataFrame object and if data is not None and originalData is not None
    if (
        isinstance(originalData, pd.DataFrame)
        and data is not None
        and isinstance(data, DataReader)
    ):
        # get inner user ids
        new_groupIds = []
        for g in groupIds:
            new_user = data.get_new_user_id(
                int(g) if isinstance(g, (int, np.integer)) else g
            )
            new_groupIds.append(new_user)
        # get inner item ids
        new_itemIds = []
        for i in itemIds:
            new_item = data.get_new_item_id(
                int(i) if isinstance(i, (int, np.integer)) else i
            )
            new_itemIds.append(new_item)
        newData = originalData.drop(
            originalData[
                (originalData.itemId.isin(new_itemIds))
                & originalData.userId.isin(new_groupIds)
            ].index
        )
        return newData
    # check if originalData is a DataReader object
    if isinstance(originalData, DataReader):
        # get inner user ids
        new_groupIds = []
        for g in groupIds:
            new_user = originalData.get_new_user_id(
                int(g) if isinstance(g, (int, np.integer)) else g
            )
            new_groupIds.append(new_user)
        # get inner item ids
        new_itemIds = []
        for i in itemIds:
            new_item = originalData.get_new_item_id(
                int(i) if isinstance(i, (int, np.integer)) else i
            )
            new_itemIds.append(new_item)
        newData = originalData.dataset.drop(
            originalData.dataset[
                (originalData.dataset.itemId.isin(new_itemIds))
                & originalData.dataset.userId.isin(new_groupIds)
            ].index
        )
        return newData


# Reads all the group ids from the file
# Input: file: the path to the file that contains the group ids
# Output: groups: a list with all the group ids
def readGroups(file):
    f = open(file, "r")
    groups = []
    for x in f:
        groups.append(x)
    return groups


# Parses through the group id and returns a list with the group members ids
# Input: group: the group id
# Output: membersIds: a list of the group members ids
def getGroupMembers(group):
    group = group.strip()
    members = group.split("_")
    membersIds = []
    for m in members:
        membersIds.append(int(m))
    return membersIds


# Returns a list of items that at least one group members has interacted with
# Input: group: a list of the group members ids
#        originalData: the original data of the system
# Output: movies: a list of distinct movie ids with which at least one group member has an interaction
def getRatedItemsByAllGroupmembers(group, originalData):
    # get original user ids
    new_group = []
    for g in group:
        new_user = originalData.get_new_user_id(
            int(g) if isinstance(g, (int, np.integer)) else g
        )
        new_group.append(new_user)

    movies = originalData.dataset[originalData.dataset.userId.isin(new_group)][
        "itemId"
    ].unique()
    original_movies = originalData.get_original_item_id(movies)

    # print the first 15 items in the list movies
    # print("movies (first 15 items): {}".format(movies[:15]))
    # print the first 15 items in the list original_movies
    # print("original_movies (first 15 items): {}".format(original_movies[:15]))

    return original_movies


# Returns all the movie ids that noone in the group has interacted with. This is used by the recommender system
# to only make predictions for new items, i.e., movies that at least one member has rated will not be suggested to the
# group
# Input: orginalData: the original data of the system
#        movie_ids: a list of all the movie ids
#        group: a list of the group members ids
# Output: movie_ids_to_pred: a list of movie ids that none of the group members has interacted with
def getMoviesForRecommendation(originalData, movie_ids, group):
    # Get a list of all movie IDs that have been watched by the group
    movie_ids_group = originalData.dataset.loc[
        originalData.dataset.userId.isin(group), "itemId"
    ]
    # Get a list off all movie IDS that that have NOT been watched by group
    movie_ids_to_pred = np.setdiff1d(movie_ids, movie_ids_group)

    return movie_ids_to_pred


def scale_predictions(
    raw_predictions,
    target_min=1,
    target_max=5,
    ref_min=0,
    ref_max=6,
    method="linear",
):
    """
    Scale raw predictions to the target range [target_min, target_max].

    Args:
        raw_predictions (array-like): The raw prediction values.
        target_min (float): Minimum of the target range (default: 1).
        target_max (float): Maximum of the target range (default: 5).
        ref_min (float): Reference minimum for raw predictions (default: 0.0).
        ref_max (float): Reference maximum for raw predictions (default: 6.0).
        method (str): Scaling method ('linear' or 'quantile').

    Returns:
        numpy.ndarray: Scaled predictions.
    """
    raw_predictions = np.array(raw_predictions)

    if len(raw_predictions) == 0:
        raise ValueError("Raw predictions array is empty.")

    if method == "linear":
        # Handle outliers using IQR
        q1, q3 = np.percentile(raw_predictions, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        clipped_predictions = np.clip(raw_predictions, lower_bound, upper_bound)

        # Compute min and max on clipped data
        min_raw = np.min(clipped_predictions)
        max_raw = np.max(clipped_predictions)

        # Scale to [target_min, target_max]
        if max_raw == min_raw:
            if ref_max == ref_min:
                scaled_value = (target_min + target_max) / 2
            else:
                scaled_value = target_min + (max_raw - ref_min) * (
                    target_max - target_min
                ) / (ref_max - ref_min)
                scaled_value = np.clip(scaled_value, target_min, target_max)
            scaled_predictions = np.full_like(raw_predictions, scaled_value)
        else:
            scaled_predictions = target_min + (raw_predictions - min_raw) * (
                target_max - target_min
            ) / (max_raw - min_raw)

    elif method == "quantile":
        # Quantile-based scaling
        ranks = stats.rankdata(raw_predictions, method="average")
        scaled_predictions = target_min + (ranks - 1) * (target_max - target_min) / (
            len(raw_predictions) - 1
        )

    else:
        raise ValueError("Invalid method. Choose 'linear' or 'quantile'.")

    # Ensure scaled predictions are within [target_min, target_max]
    scaled_predictions = np.clip(scaled_predictions, target_min, target_max)

    return scaled_predictions


# Utilizing the model that is already trained, generate predictions for movies that none of the group members has any
# interaction with. For a specific group member add their individual recommendation list into a dictionary.
# Input: model: the recommendation model that is already trained
#        user_id_str: the group member id
#        movie_ids_to_pred: a list of movie ids that none of the group members has interacted with
# Output: predictions: a dictionary containing the movied id and its corresponding prediction for the group member
#           Key: movie id
#           Value: model's prediction for that movie and group member


def generate_recommendation(model, user_id_str, movie_ids_to_pred, data):
    # Convert user_id to integer and get the new user ID
    user_id = int(user_id_str)
    new_user_id = data.get_new_user_id(user_id)
    raw_predictions = []
    # Check if the model has item_factors and if the number of items matches the dataset
    for movie_id in movie_ids_to_pred:
        movie_id = int(movie_id)  # Ensure movie_id is treated as an intege
        raw_predictions.append(model.predict(new_user_id, movie_id))

    # Ensure raw_predictions is a numpy array
    raw_predictions = np.array(raw_predictions)

    # Flatten the predictions if it's a 2D array (single user, multiple items)
    if raw_predictions.ndim == 2 and raw_predictions.shape[0] == 1:
        raw_predictions = raw_predictions.flatten()

    # Check if the length of raw_predictions matches movie_ids_to_pred
    if len(raw_predictions) != len(movie_ids_to_pred):
        raise ValueError(
            "Mismatch between predictions and movie IDs. Check the model's predict function."
        )

    # Find the minimum and maximum raw predictions
    min_raw = np.min(raw_predictions)
    max_raw = np.max(raw_predictions)

    # Apply scaling with both methods
    scaled_linear = scale_predictions(
        raw_predictions, ref_min=min_raw, ref_max=max_raw, method="linear"
    )

    # # Plot the distributions
    # plt.figure(figsize=(10, 6))
    # sns.kdeplot(scaled_linear, label='Scaled (Linear)', color='green', fill=True, alpha=0.5)
    # plt.title('Gaussian Plot for Scaled Predictions user: {}'.format(user_id_str))
    # plt.xlabel('Prediction Values')
    # plt.ylabel('Density')
    # plt.legend()
    # plt.show()

    # Convert the scaled predictions into a dictionary with movie IDs as keys
    predictions = {
        movie_id: scaled_pred
        for movie_id, scaled_pred in zip(movie_ids_to_pred, scaled_linear)
    }
    # predictions = {movie_id: scaled_pred for movie_id, scaled_pred in zip(movie_ids_to_pred, raw_predictions)}

    # Sort the predictions in descending order of scores
    sorted_predictions = {}
    for movie_id, score in predictions.items():
        # Ensure movie_id is treated as an integer
        if isinstance(movie_id, np.integer):
            movie_id = int(movie_id)
        original_id = data.get_original_item_id(movie_id)
        # Since get_original_item_id returns a single value for integer input
        sorted_predictions[int(original_id)] = score

    # Sort the predictions in descending order of scores
    sorted_predictions = dict(
        sorted(sorted_predictions.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_predictions


# The group recommender: for the group recommender model, we aggregate the prediction scores from all group members for
# an item into one group score. The aggregation function we use is the average.
# Input: predictions: a dictionary that contains the groups members and their predictions
#           Key: the user id
#           Value: a dictionary containing the predictions for that user
#                   Key: movie ids
#                    Value: the predicted score
#        groupSize: the size of the group
#        flag: an indicator that determines if the function returns the top-1 or the top-flag
#               flag = 0, return top-1
#               flag > 0, return top-flag
#               flag < 0, return the entire list
#               flag > number of distinct movies in predictions, return the entire list
def groupRecommendations(predictions, groupSize, flag):
    # print the input

    scores = {}
    for user, pred in predictions.items():
        for m in pred:
            if m in scores:
                scores[m] = scores[m] + pred[m]
            else:
                scores[m] = pred[m]
    groupPred = {}
    for m in scores:
        groupPred[m] = scores[m] / groupSize
    sorted_pred = dict(
        sorted(groupPred.items(), key=operator.itemgetter(1), reverse=True)
    )
    # If flag = 0  return only the top 1
    # else return the top-k, where k = flag
    # Print the first 100 items in the sorted list
    # print("sorted_pred: {}".format(dict(itertools.islice(sorted_pred.items(), 15))))

    if flag == 0:
        res = next(iter(sorted_pred))
    else:
        listRec = []
        i = 0
        for key in sorted_pred:
            listRec.append(key)
            i = i + 1
            if i == flag:
                break
        res = listRec
        # print('listRec: {}'.format(listRec))
    return res


# Calculates the average item intensity for a counterfactual explanation. Used in the experiments.
# Input: e: the counterfactual explanation
#        group: a list containg the group members ids
#        data: the original data in the system
# Output: explanationIntensity: the average item intensity of the counterfactual explanation
def findAverageItemIntensityExplanation(e, group, data):
    explanationIntensity = []
    groupint64 = []
    for g in group:
        new_group = data.get_new_user_id(g)
        gg = np.int64(new_group)
        groupint64.append(gg)
    for item in e:
        new_item = data.get_new_item_id(item)
        tmp = []
        tmp.append(new_item)
        intensity = len(
            data.dataset[
                (data.dataset.itemId.isin(tmp) & data.dataset.userId.isin(groupint64))
            ]
        )
        intensity = intensity / len(group)
        explanationIntensity.append(intensity)
    #
    return explanationIntensity


# Calculates the user intensity of an explanation. Used in the experiments.
# Input: e: a counterfactual explanation
#        group: a list of the group members ids
#        data: the original data of the system
# Output: userIntensity: the average user intensity of the explanation
def findUserIntensity(e, group, data):
    userIntensity = []
    new_item_exp = []
    for item in e:
        new_item = data.get_new_item_id(item)
        new_item_exp.append(new_item)
    for mm in group:
        new_user = data.get_new_user_id(mm)
        tmp = []
        tmp.append(new_user)
        intensity = len(
            data.dataset[
                (
                    data.dataset.itemId.isin(new_item_exp)
                    & (data.dataset.userId.isin(tmp))
                )
            ]
        )
        intensity = intensity / len(e)
        userIntensity.append(intensity)
    return userIntensity


# Finds the popularity of each movie (i.e., how many ratings each movie has received), and normalize that value to
# range [0,1]
# Input: movies: a list of all movie ids
#        data: the original data of the system
# Output: popMask: a dictionary containing the normalized popularity of the movies
#                   Key: movie id
#                   Value: normalized popularity
def popularityMask(movies, data):
    pop = []
    i = 0
    popMask = {}
    for m in movies:
        m = data.get_new_item_id(m)
        num = len(data.dataset[data.dataset["itemId"] == m])
        pop.append(num)
    # find the minimum value and range, and add 1% padding
    range_value = max(pop) - min(pop)
    range_value = range_value + range_value / 50
    min_value = min(pop) - range_value / 100

    # subtract the minimum value and divide by the range
    i = 0
    for m in movies:
        popMask[m] = (pop[i] - min_value) / range_value
        i = i + 1
    return popMask


# Creates a dictionary that maps each group member and the prediction score that the individual recommender system
# generated for the target item. The target item is the item for which we want to produce the explanation.
# Input: target: the target item
#        predictions: a dictionary that contains the groups members and their predictions
#            Key: the user id
#            Value: a dictionary containing the predictions for that user
#                     Key: movie ids
#                     Value: the predicted score
# Output: rel: a dictionary mapping the user id and their corresponding predicted score for the target item
#                   Key: user id
#                   Value: the user's predicted score for the target item
def relevanceMask(target, predictions):
    rel = {}
    for user, pred in predictions.items():
        if target in pred:
            rel[user] = pred[target]
            # print(rel[user])
        else:
            rel[user] = 0
    return rel


# For each item calculates the average prediction score that was generated by the individual recommender system
# for the group members. It then normalizes that sccore to the range [0,1].
# Input: m: the movie id
#        data: the original data of the system
#        relMask: the relevance Mask generated from function 'relevanceMask'
#        group: a list with the group members ids
# Output: score: the average score
def findRelevance(m, data, relMask, group):
    score = 0
    i = 0
    new_m = data.get_new_item_id(m)
    for gm in group:
        new_gm = (
            data.get_new_user_id(int(gm)) if isinstance(gm, (int, np.integer)) else gm
        )
        dd = data.dataset[
            (data.dataset["userId"] == new_gm) & (data.dataset["itemId"] == new_m)
        ]
        if dd.empty:
            continue
        score = score + relMask[gm]
        i = i + 1

    if i == 0:
        return 0
    score = score / i
    maxV = 5
    minV = 0
    range_value = maxV - minV
    range_value = range_value + range_value / 50
    min_value = minV - range_value / 100
    score = (score - min_value) / range_value
    return score


# Calculates the item intensity. Finds for a movie how many in the group have rated it. Then normalize it by the
# size of the group
# Input: m: the movie id
#        group: a list of the group members ids
#        data: the original data of the system
# Output: num: the average item intensity
def findItemIntensity(m, group, data):
    new_movie = data.get_new_item_id(m)
    new_group = []
    for g in group:
        new_group.append(data.get_new_user_id(g))

    num = len(
        data.dataset[
            (data.dataset.itemId == new_movie) & data.dataset.userId.isin(new_group)
        ]
    )
    num = num / len(group)
    return num


# Calculates the average ratings given to the movie by the group members. Then normalizes that value to [0,1]
# Input: m: the movie id
#        data: the original data of the system
#        group: a list of the group members ids
# Output: score: the average ratings given to the movie by the group members
def findRatings(m, data, group):
    new_movie = data.get_new_item_id(m)
    new_group = []
    for g in group:
        new_group.append(data.get_new_user_id(g))

    df = data.dataset[
        (data.dataset.itemId == new_movie) & data.dataset.userId.isin(new_group)
    ]
    score = df["rating"].sum()
    score = score / len(group)
    maxV = 5
    minV = 0
    range_value = maxV - minV
    range_value = range_value + range_value / 50
    min_value = minV - range_value / 100
    score = (score - min_value) / range_value
    return score


# For each item calculates a score based on the items: popularity, average predicted score, average relevance to the
# group and item intensity. Then adds these items to a dictionary and sorts it in descending order based on the scores.
# Input: itemRatedByGroup: a list of items that at least one group member has interacted with.
#        data: the original data of the system
#        members: a list of the group members ids
#        relMask: the relevance mask
#        popMask: the popularity mask
# Output: lst: the sorted list of movie ids. Sorted based on the accumulative scores of popularity, item intensity,
#              ratings and relevance
def makeChart(itemRatedByGroup, data, members, relMask, popMask):
    chart = {}
    for i in itemRatedByGroup:
        pop = popMask[i]
        itemIntensity = findItemIntensity(i, members, data)
        avgRatings = findRatings(i, data, members)
        rel = findRelevance(i, data, relMask, members)
        item_score = pop + rel + itemIntensity + avgRatings
        chart[i] = item_score

    sorted_exp = dict(sorted(chart.items(), key=operator.itemgetter(1), reverse=True))
    lst = list(sorted_exp.keys())
    # print the first 15 items in sorted_exp
    # print("sorted_exp (first 15 items): {}".format(dict(itertools.islice(sorted_exp.items(), 15))))
    # print the first 15 items in lst
    # print("lst (first 15 items): {}".format(lst[:15]))
    return lst


# Start of the experiments

# Read the ratings file.
data = DataReader(**cfg.data.test)
data.make_consecutive_ids_in_dataset()
data.binarize(binary_threshold=1)


# Train the recommendation model
algo = ALS(**cfg.model.als)
algo.fit(data)

# print("print",data.dataset.shape)


# Read the file with the group ids

group_handler = GroupInteractionHandler("../datasets/stratigis")

all_groups = group_handler.read_groups("groupsWithHighRatings5.txt")

# all_groups = readGroups("../datasets/ml-100k/groupsWithHighRatings5.txt")

movie_ids = data.dataset["itemId"].unique()
# print("number of movies: {}".format(len(movie_ids)))
# print("number of users: {}".format(len(data.dataset["userId"].unique())))


for group in all_groups:
    group = group.strip("\n")
    members = group_handler.get_group_members(group)
    # members = getGroupMembers(group)
    print(members)
    candidateMovies = getMoviesForRecommendation(data, movie_ids, members)

    predictions = {}
    for m in members:
        user_pred = generate_recommendation(algo, m, candidateMovies, data)
        predictions[m] = user_pred
    # get the group recommendation. originalGroupRec: the target item -> item we want a counterfactual explanation for
    originalGroupRec = groupRecommendations(predictions, 5, 0)
    # print('originalGroupRec_top1: {}'.format(originalGroupRec))

    found = 0

    # get all the items that at least one group member has interacted with
    rawItemRatedByGroup = group_handler.get_rated_items_by_all_groupmembers(
        members, data
    )
    # rawitemRatedByGroup = getRatedItemsByAllGroupmembers(members, data)
    itemRatedByGroup = rawItemRatedByGroup

    # check if the originalGroupRec is in the list of items that at least one group member has interacted with
    # if originalGroupRec in itemRatedByGroup:
    #     print("originalGroupRec is in itemRatedByGroup")
    # else:
    #     print("originalGroupRec is NOT in itemRatedByGroup")
    s = len(itemRatedByGroup)

    calls = 0

    explanationsFound = {}
    # Create the popularity and Relevance Mask
    popMask = popularityMask(itemRatedByGroup, data)
    relMask = relevanceMask(originalGroupRec, predictions)
    # Create the aggregated list based on the aggregation of the four metrics
    chart = makeChart(itemRatedByGroup, data, members, relMask, popMask)
    print("------------------")

    # NOTE: change the following between static size window and percentage window
    # window_size = math.floor(len(chart) * 0.1)
    window_size = 3

    print("window size: {}\tchart size: {}".format(window_size, len(chart)))
    # Create the window
    sw = SlidingWindow(chart, window_size)

    checked = []
    s = len(chart)
    if not chart:
        print("Could not find any items")
        continue
    l = 1  # noqa: E741
    exp = []
    wind_count = 0
    while True:
        # Get the sliding window
        big_window = sw.get_next_window()
        # print('big_window: {}'.format(big_window))
        # If the window has passed through all list exit with message: Could not find Explanation
        # If the explanation is found exit
        # If the group recommended system has been called more than 1000 times stop
        if big_window is None:
            print("Could not find Explanation")
            break
        if found > 0:
            break
        if calls > 1000:
            break

        # counts how many times the group recommender has been called
        calls = calls + 1
        wind_count = wind_count + 1

        # remove from the group interactions the items in the sliding window
        changedData = group_handler.create_modified_dataset(
            original_data=data.dataset,
            group_ids=members,
            item_ids=big_window,
            data=data,
        )

        # changedData = changeData(data.dataset, members, big_window, data)
        changedData1 = changedData.copy()
        # print("O primeiro item da BW está no changedData1?{}".format(changedData1.itemId.isin(big_window) & changedData1.userId.isin(members)))

        # Retrain the recommendation model
        data_retrained = DataReader(
            filepath_or_buffer=None,
            sep=None,
            names=None,
            groups_filepath=None,
            skiprows=0,
            dataframe=changedData,
        )
        data_retrained.make_consecutive_ids_in_dataset()
        data_retrained.binarize(binary_threshold=1)

        # Retrain the recommendation model
        algo_retrained = ALS(**cfg.model.als)
        algo_retrained.fit(data_retrained)

        # Get the group recommendation
        predictions_retrained = {}

        for m in members:
            user_pred = generate_recommendation(
                algo_retrained, m, candidateMovies, data_retrained
            )
            predictions_retrained[m] = user_pred
        groupRec = groupRecommendations(predictions_retrained, 5, 10)
        print("groupRec_retrained_seg (1st rec): {}".format(groupRec[0]))

        # print('groupRec_retrained: {}'.format(groupRec))

        # if the target item is still in the group recommendation list continue
        if originalGroupRec in groupRec:
            # print("originalGroupRec is in groupRec")
            continue

        else:
            # a counterfactual explanation has been found
            found = found + 1
            length = 1
            found_subset = 0
            # For the items in the sliding window find all possible combinations
            for length in range(len(big_window)):
                length = length + 1
                if found_subset > 0:
                    break
                combinations = itertools.combinations(big_window, length)
                # print('combinations: {}'.format(combinations))
                # print('length: {}'.format(length))
                for it in combinations:
                    # if a counterfactual explanation is found stop
                    # if the group recommender system has been called more than 1000 stop
                    if found_subset > 0:
                        break
                    else:
                        if calls > 1000:
                            break
                    exp = []
                    # print('it: {}'.format(it))
                    for itm in range(length):
                        exp.append(it[itm])
                    # print('exp: {}'.format(exp))
                    calls = calls + 1
                    # print('{} of {} at combination length {}'.format(l,cs,length))

                    # print(calls)
                    # print('checking set: {}'.format(exp))

                    # print('a explicacao é {} e o primeiro item da BW é {}'.format(exp,big_window[0]))

                    # Change the data and call on the group recommender system
                    changedData = group_handler.create_modified_dataset(
                        original_data=data.dataset,
                        group_ids=members,
                        item_ids=exp,
                        data=data,
                    )

                    # changedData = changeData(data.dataset, members, exp, data)
                    # print("O primeiro item da BW está no changedData (ultimo)?{}".format(changedData.itemId.isin(exp) & changedData.userId.isin(members)))
                    # print("O primeiro item da BW está no changedData1?{}".format(changedData1.itemId.isin(exp) & changedData1.userId.isin(members)))

                    # Retrain the recommendation model
                    data_retrained = DataReader(
                        filepath_or_buffer=None,
                        sep=None,
                        names=None,
                        groups_filepath=None,
                        skiprows=0,
                        dataframe=changedData,
                    )
                    data_retrained.make_consecutive_ids_in_dataset()
                    data_retrained.dataset = data_retrained.dataset.iloc[
                        1:
                    ].reset_index(drop=True)

                    # Retrain the recommendation model
                    algo_retrained = ALS(**cfg.model.als)
                    algo_retrained.fit(data_retrained)

                    predictions_retrained = {}
                    for m in members:
                        user_pred = generate_recommendation(
                            algo_retrained, m, candidateMovies, data_retrained
                        )
                        predictions_retrained[m] = user_pred
                    groupRec_last = groupRecommendations(predictions_retrained, 5, 10)
                    print("groupRec_retrained_ultimo (1st rec): {}".format(groupRec[0]))
                    # print('groupRec_retrained_again: {}'.format(groupRec))
                    if originalGroupRec in groupRec_last:
                        # print("originalGroupRec still in groupRec")
                        l = l + 1  # noqa: E741
                        continue

                    else:
                        found_subset = found_subset + 1
                        print(
                            "If the group had not interacted with these items "
                            "{}, the item of interest {} would not have appeared "
                            "on the recommendation list; instead, {} would have been recommended.".format(
                                exp, originalGroupRec, groupRec[0]
                            )
                        )
                        print("")
                        print("Explanation: {} : found at call: {}".format(exp, calls))
                        itemIntensity = findAverageItemIntensityExplanation(
                            exp, members, data
                        )
                        userIntensity = findUserIntensity(exp, members, data)
                        explanationsFound[calls] = exp
                        exp_size = len(exp)
                        print(
                            "{}\t{}\t{}\t{}".format(
                                exp_size, calls, itemIntensity, userIntensity
                            )
                        )
                    l = l + 1  # noqa: E741

            l = l + 1  # noqa: E741
    if found == 0:
        print("Explanation could not be found")
print("done")
