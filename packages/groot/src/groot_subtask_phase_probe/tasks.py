from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    slug: str
    env_name: str
    predicate_1: tuple[str, ...]
    predicate_1_description: str
    predicate_2_description: str


TASKS: dict[str, TaskSpec] = {
    "stove_moka": TaskSpec(
        slug="stove_moka",
        env_name="libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        predicate_1=("turnon", "flat_stove_1"),
        predicate_1_description="stove on",
        predicate_2_description="moka pot on stove / task success",
    ),
    "bowl_drawer": TaskSpec(
        slug="bowl_drawer",
        env_name=(
            "libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_"
            "of_the_cabinet_and_close_it"
        ),
        predicate_1=("in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"),
        predicate_1_description="bowl in bottom drawer",
        predicate_2_description="drawer closed / task success",
    ),
    "mug_microwave": TaskSpec(
        slug="mug_microwave",
        env_name=(
            "libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
        ),
        predicate_1=("in", "white_yellow_mug_1", "microwave_1_heating_region"),
        predicate_1_description="mug in microwave",
        predicate_2_description="microwave closed / task success",
    ),
}
