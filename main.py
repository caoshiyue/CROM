import os
import json

from player import *
from game import G08A
import threading
from concurrent.futures import ThreadPoolExecutor
import random
import math
# Fill in your config information to conduct experiments.

ENGINE = "xiaoai:gpt-4o-mini"  #"gpt-4o-mini", "gpt-4o" "gpt-4o-mini-2024-07-18" "deepseek-v3" "deepseek-r1"

def poisson_prob(k, lam=1.3):
    return (lam ** k * math.exp(-lam)) / math.factorial(k)

def build_player(strategy, name, persona, mean=50, std=25, player_names = [],prev_biddings=None):
    """
    Player Factory
    """
    if strategy=="agent":
        return GameAgent("llm", {"model": ENGINE},None,name=name,persona=persona,engine=ENGINE,prev_biddings=prev_biddings,)
    elif strategy=="cot":
        return CoTAgentPlayer(name, persona, ENGINE, prev_biddings)
    elif strategy=="persona":
        return PersonaAgentPlayer(name, persona, ENGINE, prev_biddings)
    elif strategy=="reflect":
        return ReflectionAgentPlayer(name, persona, ENGINE, prev_biddings)
    elif strategy=="refine":
        return SelfRefinePlayer(name, persona, ENGINE, prev_biddings)
    elif strategy=="pcot":
        return PredictionCoTAgentPlayer(name, persona, ENGINE, prev_biddings)
    elif strategy=="kr":
        return KLevelReasoningPlayer(name, persona, ENGINE,prev_biddings, player_names,level_k=2,)
    elif strategy=="spp":
        return SPPAgentPlayer(name, persona, ENGINE, prev_biddings)
    elif strategy in ["fix", "last", "mono", "monorand"]:
        return ProgramPlayer(name, strategy, mean, std)
    elif strategy=="o1" :
        return AgentPlayer(name,persona,"o3-mini")
    elif strategy=="llama" :
        return AgentPlayer(name,persona,"meta-llama/llama-3.1-8b-instruct",prev_biddings)
    elif strategy=="mistralai" :
        return AgentPlayer(name,persona,"mistralai/mixtral-8x7b-instruct",prev_biddings)
    elif strategy == "tot":
        return ToTAgentPlayer(name, persona, ENGINE, prev_biddings)
    elif strategy == "crom":
        return CROMAgent(name, persona, "openai/gpt-4o-mini", prev_biddings)
    elif strategy=="mix" :
        mix_strategies = ["cot", "tot", "kr", "reflect", "agent"]
        selected = random.choice(mix_strategies)
        
        if selected == "cot":
            return CoTAgentPlayer(name, persona, ENGINE, prev_biddings)
        elif selected == "tot":
            return ToTAgentPlayer(name, persona, ENGINE, prev_biddings)
        elif selected == "kr":
            return KLevelReasoningPlayer(name, persona, ENGINE,prev_biddings, player_names, level_k=2)
        elif selected == "reflect":
            return ReflectionAgentPlayer(name, persona, ENGINE, prev_biddings)
        elif selected == "agent":
            return AgentPlayer(name, persona, ENGINE, prev_biddings)
    else:
        raise NotImplementedError
    
def convert_biddings(biddings):
    name_map = {"Alex": "player1", "Bob": "player2", "Cindy": "player3", "David": "player4", "Eric": "player5"}
    new_biddings = {}
    for name, values in biddings.items():
        new_name = name_map.get(name, name)
        new_biddings[new_name] = values
    return new_biddings


def main(args, exp_no, prev_biddings=None):
    #Predefined Persona information
    PERSONA_A = "You are Alex and involved in a survive challenge. "
    PERSONA_B = "You are Bob and involved in a survive challenge. "
    PERSONA_C = "You are Cindy and involved in a survive challenge. "
    PERSONA_D = "You are David and involved in a survive challenge. "
    PERSONA_E = "You are Eric and involved in a survive challenge. "


    players=[]
    player_names = ["Alex", "Bob", "Cindy", "David", "Eric"]

    # build player
    A = build_player(args.player_strategy, "Alex", PERSONA_A, player_names=player_names,prev_biddings=prev_biddings)
    # Modify PlayerA's settings for ablation experiments.
    if args.player_engine: A.engine = args.player_engine
    if args.player_k:  A.k_level = args.player_k
    A.engine="xiaoai:gpt-4o-mini"
    players.append(A)

    # build opponent
    for program_name, persona in [("Bob", PERSONA_B), ("Cindy", PERSONA_C), ("David", PERSONA_D), ("Eric", PERSONA_E)]:
        players.append(build_player(args.computer_strategy, program_name, persona, args.init_mean, args.norm_std, player_names=player_names,prev_biddings=prev_biddings))

    # run multi-round game (default 10)
    Game = G08A(players)
    Game.run_multi_round(args.max_round)

    # export game records
    prefix = f"{args.player_strategy}_VS_{args.computer_strategy}_{exp_no}"
    if args.computer_strategy in ["fix", "last"]:
        prefix = f"{args.player_strategy}_VS_{args.computer_strategy}-{args.init_mean}-{args.norm_std}_{exp_no}"

    output_file = f"{args.output_dir}/{prefix}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file,"w") as fout:
        messages = {}
        biddings = {}
        logs = {}
        for agent in Game.all_players:
            if agent.is_agent:
                    messages[agent.name] = agent.message
            biddings[agent.name] = agent.biddings
            if agent.logs:
                logs[agent.name] = agent.logs

        debug_info = {
            "winners": Game.round_winner,
            "biddings": biddings,
            "message": messages,
            "logs":logs
        }

        json.dump(debug_info, fout, indent=4)
    return json.dumps(convert_biddings(biddings))


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--player_strategy', type=str, default="agent", choices=["agent","cot","pcot","kr","reflect", "persona", "refine", "spp","mk","rk","rk2","rk3","mem","o1","llama","tot","crom", "mistralai"])
    parser.add_argument('--computer_strategy', type=str, default="agent",choices=["agent", "fix", "last", "mono", "monorand","cot","pcot","kr","reflect", "persona", "refine", "spp","mk","rk","rk2","rk3","mix","mem","mix2","mix_fix","llama","tot","crom"])
    parser.add_argument("--output_dir", type=str, default="result")
    parser.add_argument("--init_mean", type=int, default=40, help="init mean value for computer player")
    parser.add_argument("--norm_std", type=int, default=5, help="standard deviation of the random distribution of computer gamers")
    parser.add_argument('--max_round', type=int, default=5)
    parser.add_argument('--start_exp', type=int, default=0)
    parser.add_argument('--exp_num', type=int, default=1)
    parser.add_argument('--player_engine', type=str, default=None, help="player's OpenAI api engine")
    parser.add_argument('--player_k', type=int, default=None, help="player's k-level (default 2)")
    parser.add_argument('--max_t', type=int, default=1, help="player's k-level (default 2)")
    args = parser.parse_args()

    #pre-play
    prev_biddings=""
    # for i in range(3):
    #     prev_biddings += main(args, 0, prev_biddings) 
    
    #fix-memory-play
    for exp_no in range(args.start_exp, args.exp_num):
        main(args, exp_no, prev_biddings)


    # threads = []
    # params = [(args, exp_no) for exp_no in range(args.start_exp, args.exp_num)]
    # with ThreadPoolExecutor(max_workers=args.max_t) as executor:
    #     for args, param in params:
    #         executor.submit(main, args, param)
    #         time.sleep(3.5)
