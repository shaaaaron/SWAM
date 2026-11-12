import sys
import os
import subprocess
import time

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback
import random
import json

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def resolve_robotwin_test_num(default=50):
    raw_value = os.environ.get("ROBOTWIN_TEST_NUM")
    if raw_value in (None, ""):
        return default

    try:
        test_num = int(raw_value)
    except ValueError as exc:
        raise ValueError("ROBOTWIN_TEST_NUM must be a positive integer") from exc

    if test_num <= 0:
        raise ValueError("ROBOTWIN_TEST_NUM must be a positive integer")
    return test_num


def build_deterministic_instruction(task_name, episode_info_list, instruction_type, episode_seed, max_descriptions):
    random_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        results = generate_episode_descriptions(task_name, episode_info_list, max_descriptions)
        candidates = results[0][instruction_type]
        if not candidates:
            raise ValueError(f"No instruction candidates found for task={task_name}, type={instruction_type}")
        instruction_idx = np.random.RandomState(episode_seed).randint(len(candidates))
        return candidates[instruction_idx]
    finally:
        random.setstate(random_state)
        np.random.set_state(np_state)


def load_episode_specs(spec_path):
    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_episode_specs(spec_path, episode_specs):
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(episode_specs, f, indent=2, ensure_ascii=False)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    run_start = time.perf_counter()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    if fixed_eval_spec_path := usr_args.get("fixed_eval_spec_path"):
        args["fixed_eval_spec_path"] = fixed_eval_spec_path

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    default_result_root = Path(__file__).resolve().parents[1] / "eval_result"
    result_root = Path(os.environ.get("ROBOTWIN_RESULT_ROOT", str(default_result_root)))
    save_dir = result_root / task_name / policy_name / task_config / ckpt_setting / current_time
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_w = int(camera_config["w"]) * 2
        video_h = int(camera_config["h"]) * 2
        video_size = f"{video_w}x{video_h}"
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = resolve_robotwin_test_num()
    print(
        f"[ROBOTWIN_EVAL] task={task_name} episodes={test_num} "
        f"instruction_type={instruction_type} fixed_eval_spec_path={args.get('fixed_eval_spec_path')!r}",
        flush=True,
    )
    topk = 1

    model_connect_start = time.perf_counter()
    model = get_model(usr_args)
    model_connect_time = time.perf_counter() - model_connect_start
    task_eval_start = time.perf_counter()
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   eval_spec_path=save_dir / "_episode_specs.json",
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type)
    task_eval_time = time.perf_counter() - task_eval_start
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    print(
        "[ROBOTWIN_TIMING_RUN] "
        f"model_connect_s={model_connect_time:.3f} "
        f"task_eval_s={task_eval_time:.3f} "
        f"total_s={time.perf_counter() - run_start:.3f}",
        flush=True,
    )
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                eval_spec_path=None,
                test_num=50,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")
    task_start = time.perf_counter()

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]
    fixed_eval_spec_path = args.get("fixed_eval_spec_path", None)
    episode_specs = load_episode_specs(fixed_eval_spec_path) if fixed_eval_spec_path else []
    collected_episode_specs = []
    collected_episode_results = []

    args["eval_mode"] = True

    expert_check_time = 0.0
    episode_timings = []
    while succ_seed < test_num:
        if episode_specs:
            if now_id >= len(episode_specs):
                raise ValueError(
                    f"Fixed eval spec only contains {len(episode_specs)} episodes, but test_num={test_num}."
                )
            now_seed = int(episode_specs[now_id]["seed"])

        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check and not episode_specs:
            try:
                expert_check_start = time.perf_counter()
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                # stack_trace = traceback.format_exc()
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print(type(e).__name__, repr(e))
                traceback.print_exc()
                continue
            finally:
                expert_check_time += time.perf_counter() - expert_check_start

        if episode_specs or (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        episode_start = time.perf_counter()
        setup_start = time.perf_counter()
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        setup_time = time.perf_counter() - setup_start
        if episode_specs:
            instruction = episode_specs[now_id]["instruction"]
        else:
            episode_info_list = [episode_info["info"]]
            instruction = build_deterministic_instruction(
                args["task_name"],
                episode_info_list,
                instruction_type,
                now_seed,
                test_num,
            )
            collected_episode_specs.append(
                {
                    "episode_id": now_id,
                    "seed": now_seed,
                    "instruction_type": instruction_type,
                    "instruction": instruction,
                }
            )
        print(
            f"[ROBOTWIN_EPISODE] task={task_name} episode={now_id + 1}/{test_num} "
            f"seed={now_seed} instruction={instruction!r}",
            flush=True,
        )
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        video_setup_start = time.perf_counter()
        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)
        video_setup_time = time.perf_counter() - video_setup_start

        succ = False
        reset_start = time.perf_counter()
        reset_func(model)
        reset_time = time.perf_counter() - reset_start
        get_obs_time = 0.0
        eval_step_time = 0.0
        rollout_steps = 0
        rollout_start = time.perf_counter()
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            get_obs_start = time.perf_counter()
            observation = TASK_ENV.get_obs()
            get_obs_time += time.perf_counter() - get_obs_start
            eval_step_start = time.perf_counter()
            eval_func(TASK_ENV, model, observation)
            eval_step_time += time.perf_counter() - eval_step_start
            rollout_steps += 1
            if TASK_ENV.eval_success:
                succ = True
                break
        rollout_time = time.perf_counter() - rollout_start
        # 结束当前 episode，并保存 policy 侧尚未落盘的产物（如预测视频）。
        reset_start = time.perf_counter()
        reset_func(model)
        reset_time += time.perf_counter() - reset_start
        # task_total_reward += TASK_ENV.episode_score
        video_finalize_start = time.perf_counter()
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()
        video_finalize_time = time.perf_counter() - video_finalize_start

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        collected_episode_results.append(
            {
                "episode_id": now_id,
                "episode_video_index": int(TASK_ENV.test_num),
                "seed": int(now_seed),
                "instruction": instruction,
                "success": bool(succ),
            }
        )

        now_id += 1
        close_env_start = time.perf_counter()
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        close_env_time = time.perf_counter() - close_env_start
        episode_total_time = time.perf_counter() - episode_start

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        episode_timing = {
            "episode_total": episode_total_time,
            "setup": setup_time,
            "video_setup": video_setup_time,
            "reset": reset_time,
            "rollout": rollout_time,
            "get_obs": get_obs_time,
            "eval_step": eval_step_time,
            "video_finalize": video_finalize_time,
            "close_env": close_env_time,
        }
        episode_timings.append(episode_timing)
        print(
            "[ROBOTWIN_TIMING_EPISODE] "
            f"episode={TASK_ENV.test_num - 1} seed={now_seed} success={int(succ)} "
            f"steps={rollout_steps} "
            + " ".join(
                f"{name}_s={duration:.3f}"
                for name, duration in episode_timing.items()
            ),
            flush=True,
        )

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    if eval_spec_path is not None and not episode_specs:
        save_episode_specs(eval_spec_path, collected_episode_specs)
    if eval_spec_path is not None and collected_episode_results:
        results_path = Path(eval_spec_path).with_name("_episode_results.json")
        save_episode_specs(results_path, collected_episode_results)

    try:
        print_timing_summary = eval_function_decorator(policy_name, "print_timing_summary")
    except AttributeError:
        print_timing_summary = None
    if print_timing_summary is not None:
        print_timing_summary(model)

    task_total_time = time.perf_counter() - task_start
    print(
        "[ROBOTWIN_TIMING_TASK] "
        f"task={task_name} episodes={len(episode_timings)} "
        f"expert_check_s={expert_check_time:.3f} total_s={task_total_time:.3f}",
        flush=True,
    )
    if episode_timings:
        for metric in episode_timings[0]:
            values = np.asarray(
                [timing[metric] for timing in episode_timings], dtype=np.float64)
            print(
                "[ROBOTWIN_TIMING_TASK] "
                f"task={task_name} metric={metric} count={values.size} "
                f"total_s={values.sum():.3f} mean_s={values.mean():.3f} "
                f"p50_s={np.percentile(values, 50):.3f} "
                f"p95_s={np.percentile(values, 95):.3f} max_s={values.max():.3f}",
                flush=True,
            )

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    mode = config.get("task_config")
    instruction_type = config.get("instruction_type")
    expected = {
        "demo_clean": "seen",
        "demo_randomized": "unseen",
    }
    if mode not in expected or instruction_type != expected[mode]:
        parser.error(
            f"Invalid pairing: task_config={mode!r}, "
            f"instruction_type={instruction_type!r}; "
            "required: demo_clean/seen or demo_randomized/unseen"
        )
    print(
        f"[ROBOTWIN_PROTOCOL] task_config={mode} "
        f"instruction_type={instruction_type}",
        flush=True,
    )

    return config


if __name__ == "__main__":
    usr_args = parse_args_and_config()

    from test_render import Sapien_TEST
    Sapien_TEST()

    main(usr_args)
