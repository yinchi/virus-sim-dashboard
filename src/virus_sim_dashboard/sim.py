"""Simulation logic for the CUH Respiratory Virus Simulation Dashboard."""

import random
import typing
from base64 import b64decode
from dataclasses import dataclass
from io import BytesIO
from itertools import chain

import pandas as pd
import reliability.Distributions as dist
import salabim as s

LOS_CAP = 100.0  # cap length of stay at 100 days for sanity


def parse_key(key: str) -> dict[str, str]:
    """Convert a string representing a patient group into a dict representation.

    The input string is expected to be in the format: "('pathway', 'outcome', 'age_group')",
    and the output will be a dict like: {"pathway": ..., "outcome": ..., "age_group": ...}.
    """
    key = key.strip("()")
    segments = [seg.strip(" '") for seg in key.split(",")]
    return {
        "pathway": segments[0],
        "outcome": segments[1],
        "age_group": segments[2],
    }


def get_all_age_groups(data: dict) -> set[str]:
    """Get a set of all age groups present in the GIM or ICU patient data.

    We expect to extract from `data` a set of dicts like:
    {'pathway': 'gim', 'outcome': 'survived', 'age_group': (0, 15)}
    from which we want to get the unique age_group values only.
    We assume that the age groups are the same across pathways and outcomes.
    """
    return set(
        chain.from_iterable([[g["age_group"] for g in lbl["groups"]] for lbl in data.values()])
    )


def parse_gim(d: dict) -> dict:
    """Parse the simulation config for GIM patients from the main store data."""
    return {
        "p_label": d["probability"],
        "groups": [parse_key(k) | v for k, v in d["label_groups"].items()],
        "dist": d["los_fit"]["distribution"],
        "params": d["los_fit"]["parameters"],
    }


def parse_pre_icu(d: dict) -> dict:
    """Parse the pre-ICU simulation config (for ICU patients) from the main store data."""
    return {
        "p_label": d["probability"],
        "groups": [parse_key(k) | v for k, v in d["label_groups"].items()],
        "prob_pre_icu": d["prob_pre_icu"],
        "dist": d["pre_icu_los_fit"]["distribution"],
        "params": d["pre_icu_los_fit"]["parameters"],
    }


def parse_icu(d: dict) -> dict:
    """Parse the main-stay simulation config for ICU patients from the main store data."""
    return {
        "p_label": d["probability"],
        "groups": [parse_key(k) | v for k, v in d["label_groups"].items()],
        "dist": d["icu_los_fit"]["distribution"],
        "params": d["icu_los_fit"]["parameters"],
    }


def parse_post_icu(d: dict) -> dict:
    """Parse the post-ICU simulation config (for ICU patients) from the main store data."""
    return {
        "p_label": d["probability"],
        "groups": [parse_key(k) | v for k, v in d["label_groups"].items()],
        "prob_post_icu": d["prob_post_icu"],
        "dist": d["post_icu_los_fit"]["distribution"],
        "params": d["post_icu_los_fit"]["parameters"],
    }


def in_label(pathway, outcome, age_group, groups):
    """Check if a given pathway, outcome, and age group combination is present in the groups."""
    for g in groups:
        if g["pathway"] == pathway and g["outcome"] == outcome and g["age_group"] == age_group:
            return g["p_in_label"]
    return None


def get_dist(dist_name, params):
    """Return a distribution object based on the given distribution name and parameters.

    Currently only supports 'Lognormal_3P' distribution, but can be extended to support more types
    as needed.
    """
    if dist_name == "Lognormal_3P":
        return dist.Lognormal_Distribution(*params)
    # TODO: add support for other distributions as needed
    raise ValueError(f"Unsupported distribution: {dist_name}")


GroupTuple = tuple[str, str, str]  # (pathway, outcome, age_group)

# Allowed distribution types for length of stay modeling (can be extended as needed)
LOSDistribution = typing.Union[dist.Lognormal_Distribution]


@dataclass
class GimLabelInfo:
    """Information about a grouping label for GIM patients.

    Includes its probability, length of stay distribution, and group probabilities.
    Each label corresponds to a single length-of-stay distribution, and may contain multiple
    groups differentiated by outcome and age group.
    """

    p_label: float
    """Probability that a patient belongs to this label, i.e. P(label).

    Probabilities across all GimLabelInfo and IcuLabelInfo instances should sum to 1.
    """

    los_dist: LOSDistribution
    """Length of stay distribution for this label."""

    groups: dict[GroupTuple, float]
    """Mapping of groups to their probabilities within this label, i.e. P(group | label)."""


class GimInfo:
    """Complete information about GIM groups and labels in the model."""

    def __init__(self):
        """Initialize empty mappings for labels and groups."""
        self.label_to_group = dict[str, GimLabelInfo]()
        """Mapping of label names to their corresponding GimLabelInfo."""

        self.group_to_label = dict[GroupTuple, str]()
        """Mapping of groups to their corresponding label names."""

    def add_label(self, label: str, p_label: float, los_dist: LOSDistribution):
        """Add a new label with its probability and length of stay distribution."""
        init_info = {"p_label": p_label, "los_dist": los_dist, "groups": dict[GroupTuple, float]()}
        self.label_to_group[label] = GimLabelInfo(**init_info)

    def add_group_to_label(self, group: GroupTuple, label: str, p_in_label: float):
        """Add a new group to a label with its probability within that label."""
        self.group_to_label[group] = label
        self.label_to_group[label].groups[group] = p_in_label


@dataclass
class IcuLabelInfo:
    """Information about a grouping label for ICU patients.

    Includes its probability, length of stay distributions, and group probabilities.
    Each label corresponds to a single length-of-stay distributions, and may contain multiple
    groups differentiated by outcome and age group.
    """

    p_label: float
    """Probability that a patient belongs to this label, i.e. P(label).

    Probabilities across all GimLabelInfo and IcuLabelInfo instances should sum to 1.
    """

    p_pre_icu: float
    """Probability that a patient in this label stays in GIM before going to ICU.

    In other words, P(pre_icu | label)."""

    p_post_icu: float
    """Probability that a patient in this label stays in GIM after going to ICU.

    In other words, P(post_icu | label)."""

    pre_icu_los_dist: LOSDistribution
    """Length of stay distribution for the pre-ICU GIM stay for this label."""

    icu_los_dist: LOSDistribution
    """Length of stay distribution for this label."""

    post_icu_los_dist: LOSDistribution
    """Length of stay distribution for the post-ICU GIM stay for this label."""

    groups: dict[GroupTuple, float]
    """Mapping of groups to their probabilities within this label, i.e. P(group | label)."""


class IcuInfo:
    """Complete information about ICU groups and labels in the model."""

    def __init__(self):
        """Initialize empty mappings for labels and groups."""
        self.label_to_group = dict[str, IcuLabelInfo]()
        """Mapping of label names to their corresponding IcuLabelInfo."""

        self.group_to_label = dict[GroupTuple, str]()
        """Mapping of groups to their corresponding label names."""

    def add_label(
        self,
        label: str,
        p_label: float,
        p_pre_icu: float,
        p_post_icu: float,
        pre_icu_los_dist: LOSDistribution,
        icu_los_dist: LOSDistribution,
        post_icu_los_dist: LOSDistribution,
    ):
        """Add a new label with its probability and length of stay distribution."""
        init_info = {
            "p_label": p_label,
            "p_pre_icu": p_pre_icu,
            "p_post_icu": p_post_icu,
            "pre_icu_los_dist": pre_icu_los_dist,
            "icu_los_dist": icu_los_dist,
            "post_icu_los_dist": post_icu_los_dist,
            "groups": dict[GroupTuple, float](),
        }
        self.label_to_group[label] = IcuLabelInfo(**init_info)

    def add_group_to_label(self, group: GroupTuple, label: str, p_in_label: float):
        """Add a new group to a label with its probability within that label."""
        self.group_to_label[group] = label
        self.label_to_group[label].groups[group] = p_in_label


@dataclass
class PatientsInfo:
    """Combined GIM and ICU patient information for the simulation model."""

    gim_info: GimInfo
    """Information about GIM patients in the model."""

    icu_info: IcuInfo
    """Information about ICU patients in the model."""


def parse_patient_config(main_store_data: dict) -> PatientsInfo:
    """Parse the patient configuration settings from the main store data."""
    gim_data = parse_gim(main_store_data["step3"]["gim"])
    pre_icu_data = parse_pre_icu(main_store_data["step3"]["pre_icu"])
    icu_data = parse_icu(main_store_data["step3"]["icu"])
    post_icu_data = parse_post_icu(main_store_data["step3"]["post_icu"])

    gim_info = GimInfo()
    for outcome in ["survived", "died"]:
        for age_group in get_all_age_groups(gim_data):
            for label, label_data in gim_data.items():
                p_label = label_data["p_label"]
                p_in_label = in_label("gim", outcome, age_group, label_data["groups"])

                # Skip if this group is not present
                if p_in_label is None or p_in_label == 0:
                    continue

                # Create the label record if it doesn't exist
                if label not in gim_info.label_to_group:
                    gim_info.add_label(
                        label=label,
                        p_label=p_label,
                        los_dist=get_dist(label_data["dist"], label_data["params"]),
                    )

                # Add the group to label mapping
                group_tuple = ("gim", outcome, age_group)
                gim_info.add_group_to_label(group=group_tuple, label=label, p_in_label=p_in_label)

    icu_info = IcuInfo()
    for outcome in ["survived", "died"]:
        for age_group in get_all_age_groups(icu_data):
            for label, label_data in icu_data.items():
                p_label = label_data["p_label"]
                p_in_label = in_label("icu", outcome, age_group, label_data["groups"])

                # Skip if this group is not present
                if p_in_label is None or p_in_label == 0:
                    continue

                pre_icu_label_data = pre_icu_data[label]
                post_icu_label_data = post_icu_data[label]

                # Create the label record if it doesn't exist
                if label not in icu_info.label_to_group:
                    icu_info.add_label(
                        label=label,
                        p_label=p_label,
                        p_pre_icu=pre_icu_label_data["prob_pre_icu"],
                        p_post_icu=post_icu_label_data["prob_post_icu"],
                        pre_icu_los_dist=get_dist(
                            pre_icu_label_data["dist"], pre_icu_label_data["params"]
                        ),
                        icu_los_dist=get_dist(label_data["dist"], label_data["params"]),
                        post_icu_los_dist=get_dist(
                            post_icu_label_data["dist"], post_icu_label_data["params"]
                        ),
                    )

                # Add the group to label mapping
                group_tuple = ("icu", outcome, age_group)
                icu_info.add_group_to_label(group=group_tuple, label=label, p_in_label=p_in_label)

    return PatientsInfo(gim_info=gim_info, icu_info=icu_info)


class RandomPatientGenerator:
    """Generates random patients based on the GIM and ICU label information.

    Not to be confused with RandomPatientsGenerator, which generates a stream of patients over time
    based on an arrival process and invokes RandomPatientGenerator to generate attributes for each
    individual patient.
    """

    def __init__(self, gim_info: GimInfo, icu_info: IcuInfo):
        self.gim_info = gim_info
        self.icu_info = icu_info

        self.dist_labels = []
        """List of (pathway, label) tuples for all labels in GIM and ICU, e.g. ('gim', '1')."""
        self.dist_weights = []
        """List of probabilities corresponding to each label in dist_labels, i.e. P(label)."""

        for k, v in {
            lbl: self.gim_info.label_to_group[lbl].p_label
            for lbl in sorted(self.gim_info.label_to_group.keys())
        }.items():
            self.dist_labels.append(("gim", k))
            self.dist_weights.append(v)
        for k, v in {
            lbl: self.icu_info.label_to_group[lbl].p_label
            for lbl in sorted(self.icu_info.label_to_group.keys())
        }.items():
            self.dist_labels.append(("icu", k))
            self.dist_weights.append(v)

    def rand_label(self):
        """Generate a random label according to the distribution of labels in the data."""
        return random.choices(self.dist_labels, self.dist_weights, k=1)[0]

    def rand_patient_profile(self):
        """Generate a random patient with a pathway, outcome, age group, and length of stay."""
        pathway, label = self.rand_label()
        if pathway == "gim":
            label_info = self.gim_info.label_to_group[label]
        else:
            label_info = self.icu_info.label_to_group[label]

        # Sample a group within the label according to P(group | label)
        groups = list(label_info.groups.keys())
        group_weights = list(label_info.groups.values())
        group = random.choices(groups, group_weights, k=1)[0]

        ret = {
            "pathway": pathway,
            "outcome": group[1],
            "age_group": group[2],
        }

        if pathway == "gim":
            # Sample a length of stay from the label's distribution
            los = label_info.los_dist.random_samples(1)[0]
            ret["length_of_stay"] = min(float(los), LOS_CAP)

        if pathway == "icu":
            prob_pre_icu = label_info.p_pre_icu
            prob_post_icu = label_info.p_post_icu
            pre_icu_los = label_info.pre_icu_los_dist.random_samples(1)[0]
            icu_los = label_info.icu_los_dist.random_samples(1)[0]
            post_icu_los = label_info.post_icu_los_dist.random_samples(1)[0]

            ret["pre_icu_los"] = (
                min(float(pre_icu_los), LOS_CAP) if random.random() < prob_pre_icu else None
            )
            ret["icu_los"] = min(float(icu_los), LOS_CAP)
            ret["post_icu_los"] = (
                min(float(post_icu_los), LOS_CAP) if random.random() < prob_post_icu else None
            )

        return ret


@dataclass
class Scenario:
    """Information about the patient arrival scenario for the simulation model."""

    dailies: pd.Series
    """Daily patient arrivals for the simulation."""

    hourlies: pd.Series
    """Hourly patient probability distribution for the simulation."""

    jitter: float
    """Jitter to add to patient arrivals to avoid deterministic arrivals."""

    def rand_hour(self):
        """Randomly sample an hour of day for a patient arrival based on the hourly distribution."""
        return int(
            random.choices(population=self.hourlies.index, weights=self.hourlies.values, k=1)[0]
        )

    def rand_arr_time(self, env: s.Environment):
        """Randomly sample a patient arrival's time-of-day.

        Our patient generation process will first determine the number of arrivals for a day;
        then for each arrival, it will generate the arrival time-of-day using
        this method.  The patient generation process generates a batch of patients at
        the start of each day (i.e. at 00:00), and then delays each patient's arrival by the
        time-of-day generated here.

        The env parameter must support the hours() method, i.e. its time_unit cannot be None.
        """
        return env.hours(self.rand_hour() + random.uniform(0, 1))


def parse_scenario_config(main_store_data: dict) -> Scenario:
    """Parse the patient arrival scenario configuration from the main store data."""
    scenario_data = main_store_data["step4"]["scenario"]

    # read dataframe, set index, and extract the 'count' column as a Series
    dailies_bytes = BytesIO(b64decode(scenario_data["dailies"]))
    dailies_df = pd.read_feather(dailies_bytes).set_index("date")
    dailies = dailies_df["count"]

    # read dataframe, set index, and extract the 'probability' column as a Series
    hourlies_bytes = BytesIO(b64decode(scenario_data["hourlies"]))
    hourlies_df = pd.read_feather(hourlies_bytes).set_index("hour")
    hourlies = hourlies_df["probability"]

    jitter = scenario_data["jitter"]
    return Scenario(dailies=dailies, hourlies=hourlies, jitter=jitter)


class Environment(s.Environment):
    """Custom environment class for our simulation."""

    def __init__(
        self, *args, scenario: Scenario, random_patient_generator: RandomPatientGenerator, **kwargs
    ):
        """Set up the environment."""
        assert "datetime0" not in kwargs.keys(), (
            "`datetime0` should not be passed as an argument to Environment() as it is "
            "computed automatically from the `scenario` argument."
        )

        super().__init__(
            *args,
            datetime0=scenario.dailies.index.min(),
            **kwargs,
        )

        self.scenario = scenario
        """The scenario defining the arrival process for patients."""

        self.random_patient_generator = random_patient_generator
        """Generates random patient attributes according to the distributions in the data."""

        self.gim_beds = s.Resource("GIM Beds", capacity=s.inf)
        """Resource representing GIM beds.

        Capacity is infinite since we are not modeling bed constraints in this model,
        but we use the resource's built-in functionality to track bed occupancy."""

        self.gim_beds_by_age_group = dict[str, s.Resource]()
        """Mapping of age groups to their corresponding GIM bed resources.

        As with self.gim_beds, capacities are infinite and we are not modeling bed constraints.
        It is assumed each patient belongs to exactly one age group, and the groups do not
        overlap, e.g. 0-15, 16-64, 65+.

        We will initialize this as empty and populate it with a salabim.Resource for each new
        age group encountered as patients are generated.
        """

        self.icu_beds = s.Resource("ICU Beds", capacity=s.inf)
        """Resource representing ICU beds.

        Capacity is infinite since we are not modeling bed constraints in this model,
        but we use the resource's built-in functionality to track bed occupancy."""

        self.icu_beds_by_age_group = dict[str, s.Resource]()
        """Mapping of age groups to their corresponding ICU bed resources.

        As with self.icu_beds, capacities are infinite and we are not modeling bed constraints.
        It is assumed each patient belongs to exactly one age group, and the groups do not
        overlap, e.g. 0-15, 16-64, 65+.

        We will initialize this as empty and populate it with a salabim.Resource for each new
        age group encountered as patients are generated.
        """

        # Start the patient generator process by creatin an instance of RandomPatientsGenerator
        RandomPatientsGenerator(
            env=self,
            scenario=self.scenario,
            random_patient_generator=self.random_patient_generator,
            jitter=self.scenario.jitter,
        )


class Patient(s.Component):
    """Represents a patient in the simulation."""

    def __init__(self, *args, env: Environment, delay: float = 0, profile: dict, **kwargs):
        """Set up the patient.

        According to salabim rules, any parameters passed to Patient() not consumed by __init__,
        i.e. `profile`, will be passed to setup().

        Patients are generated by the RandomPatientsGenerator process daily at midnight,
        then delayed by `delay` before entering the system and invoking their process() method.
        Note `delay` is a float; use env.hours() to convert from hours to the environment's time
        unit.
        """
        super().__init__(*args, env=env, delay=delay, **kwargs)

        self.profile = profile
        """Patient attributes such as pathway, outcome, age group, length of stay, etc."""

    def process(self):
        """The main process for the patient.

        This will handle the patient's bed occupancy logic and eventual departure from the hospital.
        """
        self.env: Environment  # type hint for better autocompletion

        # Ensure the age group has corresponding GIM and ICU bed resources in the environment
        age_group = self.profile["age_group"]
        if age_group not in self.env.gim_beds_by_age_group:
            self.env.gim_beds_by_age_group[age_group] = s.Resource(
                f"GIM Beds Age {age_group}", capacity=s.inf
            )
        if age_group not in self.env.icu_beds_by_age_group:
            self.env.icu_beds_by_age_group[age_group] = s.Resource(
                f"ICU Beds Age {age_group}", capacity=s.inf
            )

        # Patient flow logic based on pathway
        if self.profile["pathway"] == "gim":
            # GIM pathway
            self.request(self.env.gim_beds, self.env.gim_beds_by_age_group[age_group])
            self.hold(self.env.days(self.profile["length_of_stay"]))
            self.release()  # No arguments = release all resources held by this component

        else:  # ICU pathway
            if self.profile["pre_icu_los"] is not None:
                # Pre-ICU GIM stay
                self.request(self.env.gim_beds, self.env.gim_beds_by_age_group[age_group])
                self.hold(self.env.days(self.profile["pre_icu_los"]))
                self.release()

            # ICU stay
            self.request(self.env.icu_beds, self.env.icu_beds_by_age_group[age_group])
            self.hold(self.env.days(self.profile["icu_los"]))
            self.release()

            if self.profile["post_icu_los"] is not None:
                # Post-ICU GIM stay
                self.request(self.env.gim_beds, self.env.gim_beds_by_age_group[age_group])
                self.hold(self.env.days(self.profile["post_icu_los"]))
                self.release()


class RandomPatientsGenerator(s.Component):
    """Generates patients according to a Scenario and RandomPatientGenerator.

    The Scenario defines the arrival process for patients, while the RandomPatientGenerator defines
    how to generate patient attributes for each individual patient.
    """

    def __init__(
        self,
        *args,
        env: Environment,
        scenario: Scenario,
        random_patient_generator: RandomPatientGenerator,
        **kwargs,
    ):
        assert env.datetime0() == scenario.dailies.index.min(), (
            "Environment start time must match the start of the scenario dailies index."
        )

        assert 0 <= scenario.jitter <= 1, "Jitter must be between 0 and 1."

        super().__init__(*args, env=env, **kwargs)

        self.scenario = scenario
        """The scenario defining the arrival process for patients."""

        self.random_patient_generator = random_patient_generator
        """Generates random patient attributes according to the distributions in the data."""

        self.jitter = scenario.jitter
        """Randomness value for the number of arrivals each day.  For exactly the arrival
        numbers as specified in the scenario, set this to 0 (the default)."""

    def process(self):
        """The main process for generating patients."""
        self.env: Environment  # type hint for better autocompletion

        for _date, daily_count in self.scenario.dailies.items():
            # Add jitter to the daily count if specified
            if self.jitter > 0:
                randomized_daily_count = int(
                    daily_count * random.uniform(1 - self.jitter, 1 + self.jitter)
                )
                randomized_daily_count = max(randomized_daily_count, 0)  # ensure non-negative count

            for _ in range(randomized_daily_count):
                # Generate a random patient profile
                patient_profile = self.random_patient_generator.rand_patient_profile()

                # Create a new patient with the generated profile and a random arrival time
                # based on the scenario.  The delay will cause the patient to arrive at the
                # correct time of day according to the scenario's hourly distribution.
                #
                # The patient will invoke its own process() method upon creation,
                # which will handle its bed occupancy logic and eventual departure.
                Patient(
                    env=self.env,
                    delay=self.scenario.rand_arr_time(self.env),
                    profile=patient_profile,
                )

            # Advance time to the next day after generating patients for the current day
            self.hold(self.env.days(1))


class EnvironmentFactory:
    """Factory class to generate simulation environments based on the main store data.

    Enables us to only parse the main store data once to create the necessary components
    for patient generation and environment setup.
    """

    def __init__(self, main_store_data: dict):
        self.patients_info = parse_patient_config(main_store_data)
        self.scenario = parse_scenario_config(main_store_data)
        self.random_patient_generator = RandomPatientGenerator(
            gim_info=self.patients_info.gim_info, icu_info=self.patients_info.icu_info
        )

    def create_environment(self) -> Environment:
        """Create a new simulation environment."""
        return Environment(
            scenario=self.scenario,
            random_patient_generator=self.random_patient_generator,
        )


def sim_once(env_factory: EnvironmentFactory) -> dict:
    """Run a single simulation and return the environment with the results."""
    env = env_factory.create_environment()
    env.run()

    gim_beds_df = env.gim_beds.claimed_quantity.as_dataframe()
    gim_beds_df.columns = ["t", "occupancy"]
    
    raise NotImplementedError("sim_once is not fully implemented yet.")


def sim_multiple(env_factory: EnvironmentFactory, n_runs: int):
    """Run multiple simulations and return the aggregated results."""
    results = []
    for _ in range(n_runs):
        env = env_factory.create_environment()
        results.append(env.run())
    
    # TODO: aggregate results across runs and return them in a structured format

    raise NotImplementedError("sim_multiple is not fully implemented yet.")
