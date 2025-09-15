# Local application/library specific imports
from pygrex.config import cfg
from pygrex.data_reader import DataReader, GroupInteractionHandler
from pygrex.models.als_model import ALS
from pygrex.recommender.group_recommender import GroupRecommender
from pygrex.utils.association_rules import AssociationRules
from pygrex.explain import RuleBasedGroupRecExplainer

# Read the ratings file.
data = DataReader(**cfg.data.test)
data.make_consecutive_ids_in_dataset()
data.binarize(binary_threshold=1)
# Train the recommendation model
algo = ALS(**cfg.model.als)
algo.fit(data)

# Read the file with the group ids
group_handler = GroupInteractionHandler(**cfg.data.groups)
all_groups = group_handler.read_groups("groupsWithHighRatings5.txt")


association_rules = AssociationRules(
    data=data, min_support=0.05, min_confidence=0.3, rating_threshold=3
)
rules = association_rules.compute()
df_filtered = association_rules.get_df_filtered_by_rating_threshold()
user_history = df_filtered.groupby("userId")["itemId"].apply(set).to_dict()


for group in all_groups:
    members = group_handler.get_group_members(group)
    print(members)
    print("------------------")

    group_recommender = GroupRecommender(data)
    group_recommender.setup_recommendation(algo, members, data)  # type: ignore
    original_group_rec = group_recommender.get_group_recommendations(50)
    # get all the items that at least one group member has interacted with

    items_rated_by_group = group_handler.get_rated_items_by_all_group_members(
        members,  # type: ignore
        data,  # type: ignore
    )

    explainer = RuleBasedGroupRecExplainer(
        rules=rules,  # type: ignore
        data=data,  # type: ignore
        pool_recommendations=original_group_rec,  # type: ignore
        members=members,  # type: ignore
        user_history=user_history,
        min_members_threshold=2,
    )

    # Find explanations
    explanations = explainer.find_explanation()
    print(f"Group: {group}, Explanations: {explanations}")
    print("------------------")
