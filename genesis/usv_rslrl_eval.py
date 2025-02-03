import argparse
import os
import pickle

import torch
from usv_env_genesis import USVEnv
from rsl_rl.runners import OnPolicyRunner


def main():
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="usv_env")
    parser.add_argument("--ckpt", type=int, default=500)
    parser.add_argument("--record", action="store_true", default=False)
    args = parser.parse_args()

    # load the configurations
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    # visualize the target
    env_cfg["visualize_target"] = True
    # for video recording
    env_cfg["visualize_camera"] = args.record
    # set the max FPS for visualization
    env_cfg["max_visualize_fps"] = 60
    # show viewer
    env_cfg["show_viewer"] = True

    # create the environment
    env = USVEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )

    # policy runner
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()

    # rollout the policy
    max_sim_step = int(5 * env_cfg["episode_length_seconds"] * env_cfg["max_visualize_fps"])
    with torch.no_grad():
        if args.record:
            env.cam.start_recording()
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, _, rews, dones, infos = env.step(actions)
                env.cam.render()
            env.cam.stop_recording(save_to_filename="test_1.mp4", fps=env_cfg["max_visualize_fps"])
        else:
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, _, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()
