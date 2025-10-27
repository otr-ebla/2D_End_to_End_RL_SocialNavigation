from social_nav_env import EnvironmentConfig, SocialNavigationEnv, keyboard_control

env = SocialNavigationEnv(EnvironmentConfig())
keyboard_control(env, linear_step=0.15, angular_step=0.25)