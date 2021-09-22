import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import sys, os
sys.path.append("C:/Users/lheindrich/Documents/Scalable-discovery/metacontroller/")
os.chdir("C:/Users/lheindrich/Documents/Scalable-discovery/metacontroller/")

from utils.distributions import Categorical, Normal
from utils.mouselab_metacontroller import MouselabMeta
from metacontroller import get_action, get_subgoal_action, get_goal_action
from tqdm import tqdm

def create_env_structure(leaves):
    sigma_val = {'V1': 5, 'V2': 10, 'V3': 20, 'V4': 40, 'G1': 100, 'G2': 120, 'G3': 140, 'G4': 160, 'G5': 180}
    goal_structure = ['G1', 'G2', 'G3', 'G4', 'G5']
    x = 17
    TREE = [[(i*x)+1 for i in range(leaves)]]
    dist = ['V1']
    for i in range(leaves):
        TREE += [[(i*x)+2, (i*x)+8, (i*x)+13], [(i*x)+3, (i*x)+7], [(i*x)+4], [], [(i*x)+4], [(i*x)+4], [(i*x)+5, (i*x)+6], [(i*x)+9, (i*x)+12], [(i*x)+4], [(i*x)+4], [(i*x)+4], [(i*x)+10, (i*x)+11], [(i*x)+14, (i*x)+17], [(i*x)+4], [(i*x)+4], [(i*x)+4], [(i*x)+15, (i*x)+16]]

        g = goal_structure.pop(0)
        goal_structure.append(g)
        dist += ['V1', 'V2', 'V3', g, 'V4', 'V4', 'V3', 'V2', 'V3', 'V4', 'V4', 'V3', 'V2', 'V3', 'V4', 'V4', 'V3']
    INIT = tuple([Normal(mu=0, sigma=sigma_val[d]) for d in dist])
    return TREE, INIT

TREE, INIT = create_env_structure(10)
goal_clicks = [i for i in range(len(TREE)) if len(TREE[i]) == 0]
subtrees = [] 
for goal in goal_clicks:
    sl = list(range(goal - 3, goal + 14))
    sl.remove(goal)
    subtrees.append(sl)
subtrees = [list(range(goal - 3, goal + 14)) for goal in goal_clicks]
term_click = len(TREE)
subgoal_clicks = [i for i in range(term_click) if i not in goal_clicks]
LOW_COST = 1
HIGH_COST = 1
SWITCH_COST = 0
SEED = 0
LOW_NODES = 17
GOAL_NODES = 10
W = np.array([[0.48662932, 0, 1.77753921, 1, 1]])

conditions = {0: "No demo",
    1: "Feedback",
    2: "Non hierarchical",
    3: "Hierarchical"}

def get_subtree(node: int):
    """ Finds nodes belonging to same subtree as a given node.

    Args:
        node (int): Selected node
        goal_clicks ([type], optional): List of goal nodes

    Returns:
        [int]: All nodes in the subtree (excluding the goal node)
    """
    for subtree in subtrees:
        if node in subtree:
            for goal in goal_clicks:
                if goal in subtree:
                    return [node for node in subtree if node != goal]

def get_goal_strategy(trialdata):
    """ Checks if participant follows a goal planning strategy. This constitutes that the participant 
        first explores a number of goal nodes and then clicks in the subtree of the best discovered goal.

    Args:
        trialdata ([dict]): Response data from a single trial

    Returns:
        boolean: True if participant follows the described strategy.
    """
    queries = trialdata["queries"]["click"]["state"]["target"]
    # Replace first element with 0 for root
    ground_truth =  [0] + trialdata["stateRewards"][1:]
    # Collect the participants initial goal clicks (before clicking a subtree node)
    initial_goal_clicks = []
    for query in queries:
        if int(query) in goal_clicks:
            initial_goal_clicks.append(int(query))
        else:
            break
    # If no goal nodes are clicked the participant is not following the goal strategy
    if len(initial_goal_clicks) == 0:
        return False
    best_goal = max(initial_goal_clicks, key=lambda x: ground_truth[x])
    # Clicks that plan the path to the best goal
    subgoal_clicks = get_subtree(best_goal)
    if len(queries)<=len(initial_goal_clicks) or int(queries[len(initial_goal_clicks)]) not in subgoal_clicks:
        return False
    return True

def get_goal_switching_strategy(trialdata):
    """ Unused. Calculates which percentage of participant clicks 
    follow the correct goal switching strategy.

    Args:
        trialdata ([type]): Response data from a single trial
    """
    queries = [int(x) for x in trialdata["queries"]["click"]["state"]["target"]]
    # Replace first element with 0 for root
    ground_truth =  [0] + trialdata["stateRewards"][1:]
    if len(queries) < 2:
        return np.nan
    previous_click = queries[0]
    relevant_clicks = 0
    correct_clicks = 0
    for next_click in queries[1:]:
        if (previous_click not in goal_clicks) and (next_click != term_click):
            relevant_clicks += 1
            # Check if next tree is in the same subtree
            subtree = get_subtree(previous_click)
            if next_click in subtree:
                correct_clicks += 1
        previous_click = next_click
    if relevant_clicks == 0:
        return np.nan
    else:
        return correct_clicks / relevant_clicks

def get_goal_switching_timings(trialdata):
    """ Unused. Calculates response times for different types of participant clicks.

    Args:
        trialdata ([type]): Response data for a single trial
    """
    queries = [int(x) for x in trialdata["queries"]["click"]["state"]["target"]]
    times = trialdata["queries"]["click"]["state"]["time"]
    # Replace first element with 0 for root
    if len(queries) < 2:
        return np.nan, np.nan, np.nan, np.nan
    previous_click = queries[0]
    previous_time = times[0]
    switching_times = []
    subtree_times = []
    goal_plan_times = []
    goal_to_subtree_times = []
    for next_click, next_time in zip(queries[1:], times[1:]):
        if next_time == None or previous_time == None:
            return np.nan, np.nan, np.nan, np.nan
        delta = next_time - previous_time
        if (previous_click not in goal_clicks) and (next_click != term_click):
            subtree = get_subtree(previous_click)
            if next_click in subtree:
                subtree_times.append(delta)
            else:
                switching_times.append(delta)
        elif (previous_click in goal_clicks) and (next_click in goal_clicks):
            goal_plan_times.append(delta)
        elif (previous_click in goal_clicks) and (next_click not in goal_clicks):
            goal_to_subtree_times.append(delta)
        previous_click = next_click
        previous_time = next_time
    switching_times = np.median(switching_times)/1000 if len(switching_times) > 0 else np.nan
    subtree_times = np.median(subtree_times)/1000 if len(subtree_times) > 0 else np.nan
    goal_plan_times = np.median(goal_plan_times)/1000 if len(goal_plan_times) > 0 else np.nan
    goal_to_subtree_times = np.median(goal_to_subtree_times)/1000 if len(goal_to_subtree_times) > 0 else np.nan
    return switching_times, subtree_times, goal_plan_times, goal_to_subtree_times

def compute_click_agreement(trialdata):
    clicks = [int(x) for x in trialdata["queries"]["click"]["state"]["target"]]
    ground_truth = [0] + trialdata["stateRewards"][1:] # Replace root node with 0
    agreement = []
    goal_agreement = []
    subgoal_agreement = []
    term_agreement = []
    goal_term_agreement = []
    subgoal_term_agreement = []
    for i in range(len(clicks)+1):
        subclicks = clicks[0:i]
        if i < len(clicks):
            next_click = clicks[i]
        else:
            next_click = len(TREE) # Term action
        # Goal agreement
        if next_click in goal_clicks:
            optimal_clicks = get_goal_action(W, TREE, INIT, HIGH_COST=HIGH_COST, LOW_COST=LOW_COST, ground_truth=ground_truth, actions=subclicks, cost_function="Hierarchical")
            if next_click in optimal_clicks:
                goal_agreement.append(1)
            else:
                goal_agreement.append(0)
        elif next_click in subgoal_clicks:
            optimal_clicks = get_subgoal_action(W, TREE, INIT, action=next_click, HIGH_COST=HIGH_COST, LOW_COST=LOW_COST, ground_truth=ground_truth, actions=subclicks, disable_meta=True, cost_function="Hierarchical")
            if next_click in optimal_clicks:
                subgoal_agreement.append(1)
            else:
                subgoal_agreement.append(0)
        # Calculate overall optimal click to analyze switching and termination behavior
        optimal_clicks = get_action(W, TREE, INIT, HIGH_COST=HIGH_COST, LOW_COST=LOW_COST, SWITCH_COST=SWITCH_COST, ground_truth=ground_truth, actions=subclicks, disable_meta=True, cost_function="Hierarchical")
        # Overall agreement
        if next_click in optimal_clicks:
            agreement.append(1)
        else:
            agreement.append(0)
        # Term agreement
        if next_click == term_click:
            if next_click in optimal_clicks:
                term_agreement.append("tp")
            else:
                term_agreement.append("fp")
        # Not terminating was incorrect if termination is optimal
        else:
            if term_click in optimal_clicks:
                term_agreement.append("fn")
            else:
                term_agreement.append("tn")
        if i > 0:
            previous_click = clicks[i-1]
            # User goal termination
            if previous_click in goal_clicks and next_click not in goal_clicks:
                # Goal termination was incorrect if any goal clicks are optimal
                if any(click in goal_clicks for click in optimal_clicks):
                    goal_term_agreement.append("fp")
                else:
                    goal_term_agreement.append("tp")
            # User goal non termination
            elif previous_click in goal_clicks and next_click in goal_clicks:
                # Goal termination was correct if any goal clicks are optimal
                if any(click in goal_clicks for click in optimal_clicks):
                    goal_term_agreement.append("tn")
                else:
                    goal_term_agreement.append("fn")
            # Subgoal termination
            elif previous_click in subgoal_clicks: 
                # Check if next click belongs to the same subtree
                subtree = get_subtree(previous_click)
                # To check whether subgoal termination is correct or incorrect we need to use optimal subgoal actions instead of the overall action
                optimal_clicks = get_subgoal_action(W, TREE, INIT, action=previous_click, HIGH_COST=HIGH_COST, LOW_COST=LOW_COST, ground_truth=ground_truth, actions=subclicks, disable_meta=True, cost_function="Hierarchical")
                # Otherwise the user switched goals
                if next_click not in subtree:
                    # Subgoal termination was incorrect if any subgoal clicks are optimal
                    if any(click in subtree for click in optimal_clicks):
                        subgoal_term_agreement.append("fp")
                    else:
                        subgoal_term_agreement.append("tp")
                elif next_click in subtree:
                    # No subgoal termination was correct if any subgoal clicks are optimal
                    if any(click in subtree for click in optimal_clicks):
                        subgoal_term_agreement.append("tn")
                    else:
                        subgoal_term_agreement.append("fn")
    term_agreement_score = balanced_acc(term_agreement)
    subgoal_term_agreement_score = balanced_acc(subgoal_term_agreement)
    goal_term_agreement_score = balanced_acc(goal_term_agreement)
    goal_agreement_score = agreement_score(goal_agreement)
    subgoal_agreement_score = agreement_score(subgoal_agreement)
    total_agreement_score = agreement_score(agreement)
    assert len(agreement) == len(clicks)+1
    assert len(goal_agreement + subgoal_agreement) == len(clicks)
    return total_agreement_score, goal_agreement_score, subgoal_agreement_score, term_agreement_score, goal_term_agreement_score, subgoal_term_agreement_score


def agreement_score(scores: list):
    """ Calcualtes accuracy from a list of binary classification outcomes

    Args:
        scores (list): list of 1, 0
    """
    if len(scores) > 0:
        return np.mean(scores)
    else:
        return np.nan

def balanced_acc(scores: list):
    """ Calculates balanced accuracy from a list of classification outcomes

    Args:
        scores (list): list of "tp", "fp", "tn", "fn"
    """
    tp = sum([1 for score in scores if score == "tp"])
    fp = sum([1 for score in scores if score == "fp"])
    tn = sum([1 for score in scores if score == "tn"])
    fn = sum([1 for score in scores if score == "fn"])

    if (tp > 0 or fn > 0) and (tn > 0 or fp > 0):
        sensitivity = (tp / (tp+fn)) 
        specificity = (tn / (fp+tn)) 
        return 0.5 * (sensitivity + specificity)
    # Normal acc if balanced measure not applicable (i.e. participant only performed one action)
    elif (tp > 0 or fn > 0) or (tn > 0 or fp > 0):
        return ((tp+tn)/(tp+fn+tn+fp))
    # NaN if participant performed no applicable actions
    else:
        return np.nan

def get_expected_score(trialdata, COST=1):
    """ Calculates the expected reward for a set of participant clicks and movement actions.

    Args:
        trialdata ([type]): Participant data for a single trial.
        COST ([float], optional): Click cost. Defaults to 1.

    Returns:
        float: Expected reward score.
    """
    path = trialdata["path"]
    queries = trialdata["queries"]["click"]["state"]["target"]
    # Replace first element with 0 for root
    ground_truth =  [0] + trialdata["stateRewards"][1:]
    reward = 0
    for node in path:
        if node in queries:
            reward += ground_truth[int(node)]
        else:
            reward += 0 # Depends on experiment, in this case the expectation of all nodes is 0
    # Click cost
    reward -= len(queries) * COST
    return reward

if __name__ == "__main__":
    path = "./increasing_variance/dataclips_wvfwgoxribsfopoggmsgawkfsbkt.json"
    data = json.load(open(path))

    df_index = ["Participant", "Condition", "TrialId", "Score", "ExpectedScore", "NumClicks", 
    "TestEnv", "GoalStrategy", "ClickAgreement", 
    "GoalAgreement", "SubgoalAgreement", "TermAgreement", "GoalTermAgreement", "SubgoalTermAgreement"]
    df_data = []

    bonus_data = {}
    known_workers = []
    good_responses = 0

    # Parse raw mturk data into dataframe
    for p_index, p_data in tqdm(list(enumerate(data["values"]))):
        # Filter out empty responses
        response_data = p_data[-1]
        if response_data != None:
            p_res_obj = json.loads(response_data)
            condition = p_res_obj["condition"]
            # Obscure worker ID for publication
            worker = p_index #p_res_obj["workerId"]
            if worker in known_workers:
                print("Duplicate worker", worker)
            else: 
                known_workers.append(worker)
            p_res = p_res_obj["data"]
            # Filter our incomplete trials
            # Get last instruction index
            # Test trials start after the last instruction
            # Index of that in the experiment is variable due to repeated instructions/quiz
            instruction_index = 0
            for i in range(len(p_res)):
                if p_res[i]["trialdata"]["trial_type"] == "instructions":
                    instruction_index = i+1
            if len(p_res) > instruction_index + 15:
                good_responses += 1
                for i in range(instruction_index+1,instruction_index+16):
                    trial = p_res[i]
                    trialdata = trial["trialdata"]
                    assert trialdata["trial_type"] == "mouselab-mdp"
                    trialid = trialdata["trial_id"]
                    queries = trialdata["queries"]["click"]["state"]["target"]
                    path = trialdata["path"]
                    score = trialdata["score"]
                    trial_id = int(trialdata["trial_id"])
                    expected_score = get_expected_score(trialdata)
                    total_agreement_score, goal_agreement_score, subgoal_agreement_score, term_agreement_score, goal_term_agreement_score, subgoal_term_agreement_score = compute_click_agreement(trialdata)                    
                    goal_strategy = get_goal_strategy(trialdata) 
                    #switching_strategy = get_goal_switching_strategy(trialdata)
                    #switching_times, subtree_times, goal_plan_times, goal_to_subtree_times = get_goal_switching_timings(trialdata)
                    df_data.append([worker, condition, trialid, score, expected_score, len(queries), 
                        trial_id, goal_strategy, total_agreement_score, goal_agreement_score, subgoal_agreement_score, term_agreement_score, goal_term_agreement_score, subgoal_term_agreement_score])
            try:
                bonus_data[worker] = p_res_obj["questiondata"]["final_bonus"]
            except:
                pass
                #print("Uncompleted but usable trial for worker", worker)

    print("Good responses", good_responses)
    df = pd.DataFrame(df_data, columns=df_index)
    # Exclude participants with 0 clicks
    participants = pd.DataFrame(df[df["NumClicks"] == 0].groupby("Participant").count()["NumClicks"])
    excluded = list(participants[participants["NumClicks"]>7].index)
    df_after_exclusion = df[~df["Participant"].isin(excluded)]
    df_after_exclusion.to_csv("./increasing_variance/main_excluded.csv")
    print("Exlcuded participants", excluded)
    print(len(excluded))