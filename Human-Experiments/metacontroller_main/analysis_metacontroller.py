import json
import pandas as pd
import numpy as np
import sys, os
sys.path.append("C:/Users/lheindrich/Documents/Scalable-discovery/metacontroller/")
os.chdir("C:/Users/lheindrich/Documents/Scalable-discovery/metacontroller/")
from utils.distributions import Categorical
from metacontroller import get_action, get_subgoal_action, get_goal_action
from tqdm import tqdm
import csv

# Environment parameters
SWITCH_COST = 0 # Cost of switching goals
HIGH_COST = 10 # Cost of computing a goal
LOW_COST = 10 # Cost of computing a low level node
SEED = 0 # Fixes generated environments for training
COST_FUNC = "Basic"

TREE = [[1, 16, 31, 46], [2, 3, 4, 5], [6], [6], [7], [7], [8], [8], [9, 10, 11, 12], [13], [13], [14], [14], [15], [15], [], [17, 18, 19, 20], [21], [21], [22], [22], [23], [23], [24, 25, 26, 27], [28], [28], [29], [29], [30], [30], [], [32, 33, 34, 35], [36], [36], [37], [37], [38], [38], [39, 40, 41, 42], [43], [43], [44], [44], [45], [45], [], [47, 48, 49, 50], [51], [51], [52], [52], [53], [53], [54, 55, 56, 57], [58], [58], [59], [59], [60], [60], []]
d0 = Categorical([0])
dr = Categorical([-1500, 0], probs=[0.1, 0.9])
di = Categorical([-10, -5, 5, 10])
dg = Categorical([0, 25, 75, 100])
node_types = [di, d0, di, di, di, di, di, di, dr, di, di, di, di, di, di, dg, d0, di, di, di, di, di, di, dr, di, di, di, di, di, di, dg, d0, di, di, di, di, di, di, dr, di, di, di, di, di, di, dg, d0, di, di, di, di, di, di, dr, di, di, di, di, di, di, dg]
INIT = tuple([r for r in node_types])

W_meta = np.array([[0.45137647, 0.2288873, 9.26596405, 0.17091717, 2.24210099]])
high_risk_clicks = [8, 23, 38, 53]
goal_clicks = [15,30,45,60]
term_click = 61
subgoal_clicks = [i for i in range(term_click) if i not in goal_clicks]
COST = 10
high_risk_click_reward = -1500 * 0.1
goal_reward = 50
meta_expected_scores = [65, 45, 60, 45, 60]
hierarchical_expected_scores =  [75, 55, -1460, 55, 70]
conditions = {
    0: "NoFeedback",
    1: "GoalSwitchingDemonstration",
    2: "NoGoalSwitchingDemonstration",
    3: "FeedbackSmallEnvironment"
}

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
            optimal_clicks = get_goal_action(W_meta, TREE, INIT, HIGH_COST=HIGH_COST, LOW_COST=LOW_COST, ground_truth=ground_truth, actions=subclicks)
            if next_click in optimal_clicks:
                goal_agreement.append(1)
            else:
                goal_agreement.append(0)
        elif next_click in subgoal_clicks:
            optimal_clicks = get_subgoal_action(W_meta, TREE, INIT, action=next_click, HIGH_COST=HIGH_COST, LOW_COST=LOW_COST, ground_truth=ground_truth, actions=subclicks)
            if next_click in optimal_clicks:
                subgoal_agreement.append(1)
            else:
                subgoal_agreement.append(0)
        # Calculate overall optimal click to analyze switching and termination behavior
        optimal_clicks = get_action(W_meta, TREE, INIT, HIGH_COST=HIGH_COST, LOW_COST=LOW_COST, SWITCH_COST=SWITCH_COST, ground_truth=ground_truth, actions=subclicks)
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
                optimal_clicks = get_subgoal_action(W_meta, TREE, INIT, action=previous_click, HIGH_COST=HIGH_COST, LOW_COST=LOW_COST, ground_truth=ground_truth, actions=subclicks)
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

def get_subtree(node: int, goal_clicks=goal_clicks):
    """ Finds nodes belonging to same subtree as a given node.

    Args:
        node (int): Selected node
        goal_clicks ([type], optional): List of goal nodes

    Returns:
        [int]: All nodes in the subtree (excluding the goal node)
    """
    subtree_goal = None
    for goal in goal_clicks:
        # Smallest goal that is larger than the subtree node
        if goal >= node and (subtree_goal == None or goal < subtree_goal):
            subtree_goal = goal
    assert subtree_goal is not None
    return list(range(subtree_goal-14,subtree_goal))

def get_expected_score(trialdata, COST=COST):
    """ Calculates the expected reward for a set of participant clicks and movement actions.

    Args:
        trialdata ([type]): Participant data for a single trial.
        COST ([float], optional): Click cost. Defaults to COST.

    Returns:
        float: Expected reward score.
    """
    path = trialdata["path"]
    queries = trialdata["queries"]["click"]["state"]["target"]
    # Replace first element with 0 for root
    ground_truth =  [0] + trialdata["stateRewards"][1:]
    reward = 0
    #print(path)
    #print(high_risk_clicks)
    for node in path:
        if node in queries or (int(node) in queries):
            reward += ground_truth[int(node)]
        else:
            if (node in high_risk_clicks) or (int(node) in high_risk_clicks):
                reward += high_risk_click_reward #Probability of triggering high risk event * negative reward of high risk event
            elif (node in goal_clicks) or (int(node) in goal_clicks):
                reward += goal_reward
            else:
                reward += 0 # Depends on experiment, in this case the expectation of all other nodes is 0
    # Click cost
    reward -= len(queries) * COST
    return reward

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
    subgoal_clicks = list(range(best_goal-14,best_goal))
    if len(queries)<=len(initial_goal_clicks) or int(queries[len(initial_goal_clicks)]) not in subgoal_clicks:
        return False
    return True

if __name__ == "__main__":
    # Load dataclip
    path = os.getcwd() + "/metacontroller_main/dataclips_qjlyuocdzgjlonohmrotzgfvlhny.json"
    data = json.load(open(path))

    df_index = ["Participant", "Condition", "TrialId", "Score", "ExpectedScore", "NumClicks", "TestEnv", 
    "HighRiskClicks", "GoalStrategy", "ClickAgreement", 
    "GoalAgreement", "SubgoalAgreement", "TermAgreement", "GoalTermAgreement", "SubgoalTermAgreement"]    
    df_data = []

    bonus_data = {}
    known_workers = []
    good_responses = 0
    demographics = []

    # Parse raw mturk data into dataframe
    print("Parsing participant responses...")
    for p_index, p_data in tqdm(enumerate(data["values"])):
        # Filter out empty responses
        response_data = p_data[-1]
        if response_data != None:
            p_res_obj = json.loads(response_data)
            condition = p_res_obj["condition"]
            # Obfuscate worker ID for publishing
            worker = p_index # p_res_obj["workerId"]
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
                #print(p_res[i]["trialdata"]["trial_type"])
                if p_res[i]["trialdata"]["trial_type"] == "instructions":
                    instruction_index = i+1
            #print(instruction_index)
            if len(p_res) > instruction_index + 5 and (instruction_index>14):
                if(p_res[-2]["trialdata"]["trial_type"] == "survey-text"):
                    try: 
                        demographics.append(p_res[-2]["trialdata"]["responses"])
                    except:
                        pass
                good_responses += 1
                for i in range(instruction_index,instruction_index+5):
                    trial = p_res[i]
                    trialdata = trial["trialdata"]
                    assert trialdata["trial_type"] == "mouselab-mdp"
                    trialid = trialdata["trial_id"]
                    queries = trialdata["queries"]["click"]["state"]["target"]
                    path = trialdata["path"]
                    score = trialdata["score"]
                    trial_id = int(trialdata["trial_id"])
                    expected_score = get_expected_score(trialdata)
                    num_risk_clicks = sum([1 for el in high_risk_clicks if str(el) in queries])
                    total_agreement_score, goal_agreement_score, subgoal_agreement_score, term_agreement_score, goal_term_agreement_score, subgoal_term_agreement_score = compute_click_agreement(trialdata)
                    goal_strategy = get_goal_strategy(trialdata) 
                    df_data.append([worker, condition, trialid, score, 
                        expected_score, len(queries), trial_id, num_risk_clicks, goal_strategy, 
                        total_agreement_score, goal_agreement_score, subgoal_agreement_score, term_agreement_score, goal_term_agreement_score, subgoal_term_agreement_score])
            else:
                pass
                #if instruction_index > 14:
                #    print([p_res[i]["trialdata"]["trial_type"] for i in range(len(p_res))])
            try:
                bonus_data[worker] = max(p_res_obj["questiondata"]["final_bonus"], 0.5)
            except:
                bonus_data[worker] = 0.5
                pass

    print("Parsed", good_responses, "complete participant responses")
    df = pd.DataFrame(df_data, columns=df_index)

    # Age counter
    num_female = 0
    age = 0
    age_counter = 0
    for x in demographics:
        p = json.loads(x)
        if p["Q2"].lower().strip() in ["female", "f", "woman", "cis female", "woman (she/her)"]:
            num_female += 1
        elif p["Q2"].lower().strip() == "male" or p["Q2"].lower().strip() == "m":
            pass
        else: 
            print("unrecognized gender", p["Q2"])
        if p["Q1"].isdigit():
            age += int(p["Q1"])
            age_counter += 1
        elif p["Q1"] == "20`":
            age += 20
            age_counter += 1
        else:
            print("Unrecognized age", p["Q1"])
    print("\nNumber of female participants", num_female)
    print("Average age", age / age_counter)

    # Exclude participants with 0 clicks
    participants = pd.DataFrame(df[df["NumClicks"] == 0].groupby("Participant").count()["NumClicks"])
    excluded = list(participants[participants["NumClicks"]>2].index)
    df_after_exclusion = df[~df["Participant"].isin(excluded)]
    df_after_exclusion.to_csv(os.getcwd() + "/metacontroller_main/main_excluded.csv")
    print("\nExlcuded participants", excluded)
    print("Total excluded", len(excluded))

    print("\nAverage bonus:", np.round(np.mean(list(bonus_data.values())),2))
    print("Minimum bonus:", np.round(np.min(list(bonus_data.values())),2))
    print("Maximum bonus:", np.round(np.max(list(bonus_data.values())),2))
    print("Summed bonues", np.sum(list(bonus_data.values())))
    with open(os.getcwd() + "/metacontroller_main/bonus.csv", 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for participant, bonus in bonus_data.items():
            if bonus > 0:
                writer.writerow([participant, bonus])
