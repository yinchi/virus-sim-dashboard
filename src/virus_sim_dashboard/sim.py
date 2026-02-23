"""Simulation logic for the CUH Respiratory Virus Simulation Dashboard."""

import random
import typing
from base64 import b64decode, b64encode
from dataclasses import dataclass
from io import BytesIO
from itertools import chain

import pandas as pd
import reliability.Distributions as dist
import salabim as s

LOS_CAP = 100.0  # cap length of stay at 100 days for sanity


# region parsing
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
    # flatten the list of groups across all labels and extract the age_group values into a set
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


def get_p_in_label(
    pathway: str,
    outcome: str,
    age_group: str,
    groups: list[dict[str, typing.Any]],
) -> float | None:
    """Check if a given pathway, outcome, and age group combination is present in the groups.

    If present, return the probability of that combination within the label, i.e. P(group | label).
    If not, return None.  Note that a group may be defined with probability 0, in which case
    this function will return 0 rather than None.
    """
    for g in groups:
        if g["pathway"] == pathway and g["outcome"] == outcome and g["age_group"] == age_group:
            return g["p_in_label"]
    return None


GroupTuple = tuple[str, str, str]  # (pathway, outcome, age_group)

# Allowed distribution types for length of stay modeling (can be extended as needed)
LOSDistribution = typing.Union[dist.Lognormal_Distribution]


def get_dist(dist_name, params) -> LOSDistribution:
    """Return a distribution object based on the given distribution name and parameters.

    Currently only supports 'Lognormal_3P' distribution, but can be extended to support more types
    as needed.
    """
    if dist_name == "Lognormal_3P":
        return dist.Lognormal_Distribution(*params)
    # TODO: add support for other distributions as needed
    raise ValueError(f"Unsupported distribution: {dist_name}")


# endregion


# region class definitions
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

    @classmethod
    def from_main_store(cls, main_store_data: dict) -> typing.Self:
        """Parse the patient configuration settings from the dashboard's main store data."""
        fit_results = main_store_data["step3"]["los_fit_results"]

        gim_data = {k: parse_gim(v) for k, v in fit_results["gim"].items()}
        pre_icu_data = {k: parse_pre_icu(v) for k, v in fit_results["icu"].items()}
        icu_data = {k: parse_icu(v) for k, v in fit_results["icu"].items()}
        post_icu_data = {k: parse_post_icu(v) for k, v in fit_results["icu"].items()}

        gim_info = GimInfo()
        for outcome in ["survived", "died"]:
            for age_group in get_all_age_groups(gim_data):
                for label, label_data in gim_data.items():
                    p_label = label_data["p_label"]
                    p_in_label = get_p_in_label("gim", outcome, age_group, label_data["groups"])

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
                    gim_info.add_group_to_label(
                        group=group_tuple, label=label, p_in_label=p_in_label
                    )

        icu_info = IcuInfo()
        for outcome in ["survived", "died"]:
            for age_group in get_all_age_groups(icu_data):
                for label, label_data in icu_data.items():
                    p_label = label_data["p_label"]
                    p_in_label = get_p_in_label("icu", outcome, age_group, label_data["groups"])

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
                    icu_info.add_group_to_label(
                        group=group_tuple, label=label, p_in_label=p_in_label
                    )

        return cls(gim_info=gim_info, icu_info=icu_info)


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

    @classmethod
    def from_main_store(cls, main_store_data: dict) -> typing.Self:
        """Parse the patient arrival scenario configuration from the main store data."""
        scenario_data = main_store_data["step4"]

        # read dataframe, set index, and extract the 'count' column as a Series
        dailies_bytes = BytesIO(b64decode(scenario_data["dailies"]))
        dailies_df = pd.read_feather(dailies_bytes).set_index("date")
        dailies = dailies_df["count"]

        # read dataframe, set index, and extract the 'probability' column as a Series
        hourlies_bytes = BytesIO(b64decode(scenario_data["hourlies"]))
        hourlies_df = pd.read_feather(hourlies_bytes).set_index("hour")
        hourlies = hourlies_df["probability"]

        jitter = scenario_data["jitter"]
        return cls(dailies=dailies, hourlies=hourlies, jitter=jitter)


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
            random_seed="*",  # use a random seed for each simulation run
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

    def process(self) -> None:
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

    def process(self) -> None:
        """The main process for generating patients."""
        self.env: Environment  # type hint for better autocompletion

        for _date, daily_count in self.scenario.dailies.items():
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
        self.patients_info = PatientsInfo.from_main_store(main_store_data)
        self.scenario = Scenario.from_main_store(main_store_data)
        self.random_patient_generator = RandomPatientGenerator(
            gim_info=self.patients_info.gim_info, icu_info=self.patients_info.icu_info
        )

    def create_environment(self) -> Environment:
        """Create a new simulation environment."""
        return Environment(
            scenario=self.scenario,
            random_patient_generator=self.random_patient_generator,
        )


# endregion


# region simulation
@dataclass
class SimResult:
    """Dataclass to store the result of a single simulation run."""

    gim: pd.DataFrame
    icu: pd.DataFrame


@dataclass
class SimMultipleResult:
    """Dataclass to store the result of multiple simulation runs."""

    gim: list[pd.DataFrame]
    icu: list[pd.DataFrame]

    def to_dict(self) -> dict:
        """Convert the simulation results to a JSON-serializable format."""

        def df_to_str(df: pd.DataFrame) -> dict:
            bytes_io = BytesIO()
            df.to_feather(bytes_io)
            return {"data": b64encode(bytes_io.getvalue()).decode("utf-8")}

        return {
            "gim": [df_to_str(df) for df in self.gim],
            "icu": [df_to_str(df) for df in self.icu],
        }

    @classmethod
    def from_dict(cls, json_data: dict) -> typing.Self:
        """Create a SimMultipleResult instance from JSON data."""

        def str_to_df(data_str: dict) -> pd.DataFrame:
            bytes_io = BytesIO(b64decode(data_str["data"]))
            return pd.read_feather(bytes_io)

        gim = [str_to_df(df_str) for df_str in json_data["gim"]]
        icu = [str_to_df(df_str) for df_str in json_data["icu"]]
        return cls(gim=gim, icu=icu)


def sim_once(env_factory: EnvironmentFactory) -> SimResult:
    """Run a single simulation and return the daily maximum occupancy for GIM and ICU beds."""
    env = env_factory.create_environment()
    env.run()

    # Daily max GIM occupancy
    gim_beds_by_age_group_df = {}
    gim_beds_by_age_group_df_daily = {}

    for age_group in sorted(env.gim_beds_by_age_group.keys()):
        gim_beds_by_age_group_df[age_group] = env.gim_beds_by_age_group[
            age_group
        ].claimed_quantity.as_dataframe()
        gim_beds_by_age_group_df[age_group].columns = ["t", f"{age_group}"]
        gim_beds_by_age_group_df[age_group].t = gim_beds_by_age_group_df[age_group].t.map(
            env.t_to_datetime
        )

        gim_beds_by_age_group_df_daily[age_group] = (
            gim_beds_by_age_group_df[age_group].resample("D", on="t").max().ffill()
        )

    gim_beds_summary_df = (
        pd.concat(gim_beds_by_age_group_df_daily.values(), axis=1, sort=True).ffill().fillna(0)
    )

    # Daily max ICU occupancy
    icu_beds_by_age_group_df = {}
    icu_beds_by_age_group_df_daily = {}

    for age_group in sorted(env.icu_beds_by_age_group.keys()):
        icu_beds_by_age_group_df[age_group] = env.icu_beds_by_age_group[
            age_group
        ].claimed_quantity.as_dataframe()
        icu_beds_by_age_group_df[age_group].columns = ["t", f"{age_group}"]
        icu_beds_by_age_group_df[age_group].t = icu_beds_by_age_group_df[age_group].t.map(
            env.t_to_datetime
        )

        icu_beds_by_age_group_df_daily[age_group] = (
            icu_beds_by_age_group_df[age_group].resample("D", on="t").max().ffill()
        )

    icu_beds_summary_df = (
        pd.concat(icu_beds_by_age_group_df_daily.values(), axis=1, sort=True).ffill().fillna(0)
    )

    return SimResult(
        gim=gim_beds_summary_df,
        icu=icu_beds_summary_df,
    )


def sim_multiple(
    env_factory: EnvironmentFactory,
    n_runs: int,
    set_progress: typing.Callable[[tuple[str, float]], None],  # progress_text, progress_pct
) -> SimMultipleResult:
    """Run multiple simulations and return the aggregated results."""
    results_gim = []
    results_icu = []
    for i in range(n_runs):
        result = sim_once(env_factory)
        results_gim.append(result.gim)
        results_icu.append(result.icu)
        set_progress((f"Running: {i + 1}/{n_runs} iterations", (i + 1) / n_runs * 100))

    return SimMultipleResult(gim=results_gim, icu=results_icu)


def get_quantiles(result_list: list[pd.DataFrame], groupings: list[str]) -> pd.DataFrame:
    """Get daily quantiles of bed occupancy for the selected age groups.

    The `result_list` is a list of dataframes for GIM or ICU occupancy, e.g. `result.gim` or
    `result.icu` where result is the result of `sim_multiple()`.

    The quantiles returned are the 10th, 25th, 50th, 75th, and 90th percentiles
    (lower and upper deciles and quartiles, and the median).
    """
    return (
        pd.concat([result.loc[:, groupings].sum(axis=1) for result in result_list], axis=1)
        .fillna(0)
        .quantile([0.1, 0.25, 0.5, 0.75, 0.9], axis=1)
        .T
    )


# endregion
