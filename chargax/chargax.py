from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
import jaxnasium as jym
import numpy as np
from jaxnasium import TimeStep
from jaxtyping import Array, Float, PRNGKeyArray

from chargax._util import year_and_doy

from ._default_data_loaders import (
    build_default_grid_price_fn,
    build_default_scenario,
    build_leave_cars_fn,
)
from ._station_layout import EVSE, ChargingStation, StationBattery, _PassiveNode


class EnvState(jym.EnvState):
    grid: ChargingStation
    datetime: jdt.Datetime
    elec_customer_sell_price: float
    timestep: int = 0

    @property
    def year_and_doy(self):
        return year_and_doy(self.datetime)

    @property
    def year(self):
        return self.year_and_doy[0]

    @property
    def day_of_year(self):
        return self.year_and_doy[1]

    @property
    def day_of_week(self):
        """0-6 for Monday-Sunday. 1970-01-01 was a Thursday (3), which is our reference."""
        return (self.day_of_year + 3) % 7

    @property
    def is_workday(self) -> bool:
        """
        Determine if the current day is a workday (Monday to Friday).
        """
        return self.day_of_week < 5

    # Reward variables:
    profit: float = 0.0
    uncharged_percentages: float = 0.0
    uncharged_kw: float = 0.0
    charged_overtime: int = 0
    charged_undertime: int = 0
    rejected_customers: int = 0
    served_customers: int = 0
    customer_satisfaction: float = 0.0  # unused in base version
    exceeded_capacity: float = 0.0
    total_charged_kw: float = 0.0
    total_discharged_kw: float = 0.0

    # New Normalized satisfaction reward variables
    sat_uncharged_norm: float = 0.0
    sat_overtime_norm: float = 0.0
    sat_undertime_norm: float = 0.0

    # Customer fairness bookkeeping (per scored customer, s_i in [0, 1]).
    # Per-group arrays: index 0 = time-sensitive, index 1 = charge-sensitive
    fairness_cost: float = 0.0  # sum_i (1 - s_i)^2  (reward penalty / CMDP cost)
    group_served: Array = eqx.field(default_factory=lambda: jnp.zeros(2))
    group_s_sum: Array = eqx.field(default_factory=lambda: jnp.zeros(2))
    group_s_sq_sum: Array = eqx.field(default_factory=lambda: jnp.zeros(2))
    group_unfair: Array = eqx.field(default_factory=lambda: jnp.zeros(2))
    group_min_s: Array = eqx.field(default_factory=lambda: jnp.ones(2))


class Chargax(jym.Environment):
    station: ChargingStation
    """The charging station layout defining EVSEs, batteries, and power limits."""

    simulation_starting_year: int = 2024
    """Calander year for the simulation. Default reset() will randomly sample a day within this year and set
    the datetime accordingly in the state. This may then be used to query time-dependent data."""

    elec_customer_sell_price: float = 0.75  # €/kWh
    """Price in €/kWh charged to customers for electricity delivered. This value is also set on the state and
    therefor may be dynamically adjusted."""

    get_cars_departing: Callable[[PRNGKeyArray, EVSE], Array] = build_leave_cars_fn()
    """Callable that determines which cars leave at each timestep given RNG and EVSE state."""

    get_num_cars_arriving: Callable[[PRNGKeyArray, EnvState], int] = None
    """Callable that returns the number of new cars arriving given RNG and environment state."""

    get_new_cars_arriving: Callable[[PRNGKeyArray, EnvState], EVSE] = None
    """Callable that generates EVSE entries for newly arriving cars given RNG and environment state."""

    get_grid_buy_price: Callable[[EnvState], float] = None
    """Callable that returns the grid electricity buy price in €/kWh for the current state."""

    get_grid_sell_price: Callable[[EnvState], float] = None
    """Callable that returns the grid electricity sell price in €/kWh for the current state."""

    # reward alpha values
    capacity_exceeded_alpha: float = 0.0
    """Reward penalty weight for exceeding the station's grid capacity limit."""

    charged_satisfaction_alpha: float = 0.0
    """Reward penalty weight for unmet customer charging demand (uncharged kWh)."""

    time_satisfaction_alpha: float = 0.0
    """Reward penalty weight for overtime/undertime relative to customer departure."""

    rejected_customers_alpha: float = 0.0
    """Reward penalty weight for each customer rejected due to no available charger."""

    battery_degradation_alpha: float = 0.0
    """Reward penalty weight for battery degradation, proxied by total discharged kWh."""

    beta: float = 0.0
    """Discount factor applied to early-departure (undertime) within the time satisfaction penalty."""

    # new reward normalization alpha values
    norm_satisfaction_alpha: float = 0.0
    norm_w_uncharged: float = 1.0
    norm_w_overtime: float = 1.0
    norm_w_undertime: float = -1.0

    # customer fairness
    fairness_alpha: float = 0.0
    """Penalty weight (or Lagrange multiplier) on the quadratic fairness cost."""
    fairness_threshold: float = 0.8
    """Satisfaction level below which a customer counts as 'unfairly served'."""

    # Env options:
    num_discretization_levels: int = 10
    """Number of discrete action levels per charger (e.g. 10 means 10%, 20%, … of max rate)."""

    minutes_per_timestep: int = 5
    """Duration of each simulation timestep in minutes."""

    simulation_length_days: int = 1
    """Number of days to simulate per episode. The environment will terminate after this many days."""

    renormalize_currents: bool = True
    """Whether to redistribute currents across chargers to respect shared capacity constraints."""

    allow_discharging: bool = True
    """Whether vehicle-to-grid discharging (negative current) is permitted."""

    price_hour_lookahead: int = 6
    """Number of future hours of electricity prices included in the observation."""

    default_data_kwargs: Dict = eqx.field(static=True, default_factory=lambda: {})
    """Keyword arguments passed to default data loaders for car/price scenario configuration."""

    @property
    def max_episode_steps(self) -> int:
        return int(24 * 60 // self.minutes_per_timestep) * self.simulation_length_days

    @property
    def includes_battery(self) -> bool:
        return bool(self.station.batteries)

    def __post_init__(self):
        if self.get_num_cars_arriving is None or self.get_new_cars_arriving is None:
            car_profile = self.default_data_kwargs.get("car_profile", "eu")
            user_profile = self.default_data_kwargs.get("user_profile", "highway")
            average_cars_per_day = self.default_data_kwargs.get(
                "average_cars_per_day", "high"
            )
            get_num_cars, get_new_cars = build_default_scenario(
                self,
                car_profile=car_profile,
                user_profile=user_profile,
                average_cars_per_day=average_cars_per_day,
            )
            if self.get_num_cars_arriving is None:
                self.__setattr__("get_num_cars_arriving", get_num_cars)
            if self.get_new_cars_arriving is None:
                self.__setattr__("get_new_cars_arriving", get_new_cars)

        if self.get_grid_buy_price is None or self.get_grid_sell_price is None:
            grid_price_dataset = self.default_data_kwargs.get(
                "grid_price_dataset", "2023_NL"
            )
            sell_price_margin = self.default_data_kwargs.get("grid_sell_margin", -0.03)
            if self.get_grid_buy_price is None:
                self.__setattr__(
                    "get_grid_buy_price",
                    build_default_grid_price_fn(
                        self, dataset=grid_price_dataset, offset=0
                    ),
                )

            if self.get_grid_sell_price is None:
                self.__setattr__(
                    "get_grid_sell_price",
                    build_default_grid_price_fn(
                        self, dataset=grid_price_dataset, offset=sell_price_margin
                    ),
                )

    def reset_env(self, key: PRNGKeyArray) -> Tuple[Dict[str, Array], EnvState]:

        random_day_of_year = jax.random.randint(key, (), 0, 365)
        year = self.simulation_starting_year
        random_day = jdt.to_datetime(f"{int(year)}-01-01") + jdt.Timedelta(
            days=random_day_of_year
        )
        state = EnvState(
            datetime=random_day,
            grid=self.station,
            elec_customer_sell_price=self.elec_customer_sell_price,
        )
        state = self.set_passive_throughputs(state)
        observation = self.get_observation(state)
        return observation, state

    def step_env(
        self, rng: PRNGKeyArray, old_state: EnvState, actions: Dict[str, Array]
    ) -> Tuple[TimeStep, EnvState]:
        key1, key2 = jax.random.split(rng)
        new_state = old_state

        new_state = self.set_charging_currents(new_state, actions)

        charging_ports = new_state.grid.evses_flat
        batteries = new_state.grid.batteries_flat

        new_state, charging_ports, batteries = self.charge_cars_and_update_batteries(
            new_state, charging_ports, batteries
        )
        new_state, charging_ports = self.update_time_and_clear_cars(
            key1, new_state, charging_ports
        )
        new_state, charging_ports = self.add_new_cars(key2, new_state, charging_ports)
        new_state = self.score_cars_at_end_of_day(
            new_state,
            charging_ports,
            is_last_step=(old_state.timestep + 1) >= self.max_episode_steps,
        )

        # Zero dynamic state for disconnected ports; preserve charger config
        mask = charging_ports.charger_is_car_connected
        config = {
            name: getattr(charging_ports, name)
            for name in (
                "voltage",
                "max_current",
                "max_kw_throughput",
                "efficiency",
                "cumulative_efficiency",
            )
        }
        charging_ports = jax.tree.map(lambda p: p * mask, charging_ports)
        charging_ports = charging_ports.replace(**config)

        updated_grid = new_state.grid.update_evses_from_flat(
            charging_ports
        ).update_batteries_from_flat(batteries)

        timestep_elapsed_time = jdt.Timedelta(seconds=self.minutes_per_timestep * 60)
        new_state = new_state._replace(
            grid=updated_grid,
            timestep=old_state.timestep + 1,
            datetime=old_state.datetime + timestep_elapsed_time,
        )

        timestep_object = jym.TimeStep(
            observation=self.get_observation(new_state),
            reward=self.get_reward(old_state, new_state),
            terminated=self.get_terminated(new_state),
            truncated=self.get_truncated(new_state),
            info=self.get_info(new_state, actions, old_state=old_state),
        )

        return timestep_object, new_state

    def set_passive_throughputs(self, state: EnvState) -> EnvState:
        """Set uncontrollable passive loads from their load profiles (kW rate for this step)."""

        def _passive_throughput(passive: _PassiveNode) -> _PassiveNode:
            load_kw = passive.get_current_load(state)
            return passive.replace(
                throughput_now_kw=jnp.asarray(load_kw, dtype=jnp.float32)
            )

        if not state.grid.passives:
            return state

        new_passives = jax.tree.map(
            _passive_throughput,
            state.grid.passives,
            is_leaf=lambda x: isinstance(x, _PassiveNode),
        )
        return state._replace(grid=state.grid.update_passives_from_list(new_passives))

    def set_charging_currents(self, state: EnvState, actions: Array) -> EnvState:
        """Set new currents and power levels based on actions"""
        state = self.set_passive_throughputs(state)

        def _evse_action(evse: EVSE, action: Array) -> EVSE:
            if self.allow_discharging:
                action = action - 1
            current = jnp.clip(
                action * evse.max_current,
                -evse.car_max_current_outtake if self.allow_discharging else 0,
                evse.car_max_current_intake,
            )
            return evse.replace(charger_current_now=current)

        def _battery_action(battery: StationBattery, action: Array) -> StationBattery:
            action = action - 1
            desired_output_kw = action * battery.max_kw_throughput
            desired_output_kw_now = self.kw_to_kw_this_timestep(desired_output_kw)
            new_desired_battery_level = jnp.clip(
                battery.battery_now + desired_output_kw_now, 0, battery.capacity_kw
            )
            battery_change = new_desired_battery_level - battery.battery_now
            actual_output_kw = battery_change * (60 / self.minutes_per_timestep)
            return battery.replace(throughput_now_kw=actual_output_kw)

        actions = jax.tree.map(lambda x: x / self.num_discretization_levels, actions)

        new_evses = jax.tree.map(
            _evse_action,
            state.grid.evses,
            actions["evses"],
            is_leaf=lambda x: isinstance(x, EVSE),
        )
        updated_grid = state.grid.update_evses_from_list(new_evses)
        if "batteries" in actions:
            new_batteries = jax.tree.map(
                _battery_action,
                state.grid.batteries,
                actions["batteries"],
                is_leaf=lambda x: isinstance(x, StationBattery),
            )
            updated_grid = updated_grid.update_batteries_from_list(new_batteries)

        if self.renormalize_currents:
            updated_grid = updated_grid.distribute()

        exceeded_capacity = updated_grid.exceeded_power_all_children

        return state._replace(
            grid=updated_grid,
            exceeded_capacity=state.exceeded_capacity + exceeded_capacity,
        )

    def charge_cars_and_update_batteries(
        self, state: EnvState, charging_ports: EVSE, batteries: StationBattery
    ) -> tuple[EnvState, EVSE]:

        # (dis)charge cars:
        charging_now = self.kw_to_kw_this_timestep(charging_ports.power_output)
        previous_battery = charging_ports.car_battery_now_kw
        new_battery = (previous_battery + charging_now).clip(
            charging_ports.car_arrival_battery_kw,  # can't discharge under arrival battery
            charging_ports.car_battery_capacity_kw,
        )
        real_charged_this_timestep = new_battery - previous_battery

        # (dis)charge station batteries:
        batteries_throughput_now_kw = self.kw_to_kw_this_timestep(
            batteries.throughput_now_kw
        )
        new_station_battery_level = jnp.clip(
            batteries.battery_now + batteries_throughput_now_kw,
            0,
            batteries.capacity_kw,
        )
        batteries = batteries.replace(battery_now=new_station_battery_level)

        # Calculate customer revenue (EVSEs only)
        energy_sold = jnp.maximum(
            jnp.maximum(real_charged_this_timestep, 0.0)
            - charging_ports.car_discharged_this_session_kw,
            0.0,
        ).sum()
        revenue = energy_sold * state.elec_customer_sell_price
        discharged_this_session = (
            charging_ports.car_discharged_this_session_kw + -real_charged_this_timestep
        ).clip(0)

        grid_draw_evses = jnp.where(
            real_charged_this_timestep >= 0,
            real_charged_this_timestep / charging_ports.cumulative_efficiency,
            real_charged_this_timestep * charging_ports.cumulative_efficiency,
        )
        grid_draw_batteries = jnp.where(
            batteries_throughput_now_kw >= 0,
            batteries_throughput_now_kw / batteries.cumulative_efficiency,  # charging
            batteries_throughput_now_kw
            * batteries.cumulative_efficiency,  # discharging
        )
        # NOTE: The draw / supply of passives is not included.
        total_grid_draw = grid_draw_evses.sum() + grid_draw_batteries.sum()
        elec_price = jax.lax.select(
            total_grid_draw >= 0,
            self.get_grid_buy_price(state),
            self.get_grid_sell_price(state),
        )
        profit = state.profit + revenue - total_grid_draw * elec_price
        charging_ports = charging_ports.replace(
            car_discharged_this_session_kw=discharged_this_session,
            car_battery_now_kw=new_battery,
        )
        total_charged = jnp.maximum(real_charged_this_timestep, 0.0).sum()
        total_discharged = jnp.maximum(-real_charged_this_timestep, 0.0).sum()

        return (
            state._replace(
                profit=profit,
                total_charged_kw=total_charged + state.total_charged_kw,
                total_discharged_kw=total_discharged + state.total_discharged_kw,
            ),
            charging_ports,
            batteries,
        )

    def update_time_and_clear_cars(
        self, key: PRNGKeyArray, state: EnvState, ports: EVSE
    ) -> tuple[EnvState, EVSE]:
        new_time_till_leave = ports.car_time_till_leave - self.minutes_per_timestep
        new_time_waited = ports.car_time_waited + self.minutes_per_timestep

        ports = ports.replace(
            car_time_till_leave=new_time_till_leave.astype(int),
            car_time_waited=new_time_waited,
        )

        cars_leaving = self.get_cars_departing(key, ports)
        cars_leaving = (cars_leaving * ports.charger_is_car_connected).astype(
            bool
        )  # Only consider connected cars for leaving

        state = self.set_customer_satisfaction_values(state, ports, cars_leaving)

        ports = ports.replace(
            charger_is_car_connected=ports.charger_is_car_connected * ~cars_leaving,
        )

        return state, ports

    def set_customer_satisfaction_values(
        self, state: EnvState, ports: EVSE, cars_leaving: Array
    ) -> EnvState:
        uncharged_percentages = (
            cars_leaving * jnp.maximum(ports.car_battery_desired_remaining, 0)
        ).sum()
        uncharged_kw = (
            cars_leaving * jnp.maximum(0, ports.car_battery_desired_remaining_kw)
        ).sum()
        charged_overtime = (
            jnp.abs(cars_leaving * jnp.minimum(0, ports.car_time_till_leave))
            .sum()
            .astype(int)
        )  # Use previous time till leave to calculate overtime
        charged_undertime = (
            (cars_leaving * jnp.maximum(0, ports.car_time_till_leave)).sum().astype(int)
        )
        num_cars_leaving = cars_leaving.sum()


        # New Normalized satisfaction reward values
        leaving = cars_leaving.astype(jnp.float32)
        charge_sensitive = ports.charge_sensitive.astype(jnp.float32)
        planned_stay = jnp.maximum(ports.car_time_waited + ports.car_time_till_leave, self.minutes_per_timestep)

        uncharged_n = leaving * jnp.clip(ports.car_battery_desired_remaining, 0.0, 1.0)
        overtime_n = leaving * charge_sensitive * jnp.clip(-ports.car_time_till_leave / planned_stay, 0.0, 1.0)
        undertime_n = leaving * charge_sensitive * jnp.clip(ports.car_time_till_leave / planned_stay, 0.0, 1.0)

        state = self.update_fairness_values(state, ports, cars_leaving)
        return state._replace(
            uncharged_percentages=state.uncharged_percentages + uncharged_percentages,
            uncharged_kw=state.uncharged_kw + uncharged_kw,
            charged_overtime=state.charged_overtime + charged_overtime,
            charged_undertime=state.charged_undertime + charged_undertime,
            served_customers=state.served_customers + num_cars_leaving,
            sat_uncharged_norm=state.sat_uncharged_norm + uncharged_n.sum(),
            sat_overtime_norm=state.sat_overtime_norm + overtime_n.sum(),
            sat_undertime_norm=state.sat_undertime_norm + undertime_n.sum(),
        )

    # ------------------------------------------------------------------
    # Customer fairness
    # ------------------------------------------------------------------
    def customer_satisfaction(self, ports: EVSE) -> Array:
        """Per-customer satisfaction s_i in [0, 1] for every charger slot.

        Time-sensitive customers (leave at a set time): share of the requested
        energy that was delivered. Charge-sensitive customers (stay until their
        target charge): planned stay / actual stay, i.e. 1 without overtime.
        """
        requested_kw = (
            ports.car_desired_battery_percentage * ports.car_battery_capacity_kw
            - ports.car_arrival_battery_kw
        )
        delivered_kw = ports.car_battery_now_kw - ports.car_arrival_battery_kw
        s_energy = jnp.where(
            requested_kw > 1e-6,
            jnp.clip(delivered_kw / jnp.maximum(requested_kw, 1e-6), 0.0, 1.0),
            1.0,
        )
        planned_stay = ports.car_time_waited + ports.car_time_till_leave
        s_time = jnp.clip(
            planned_stay / jnp.maximum(ports.car_time_waited, 1), 0.0, 1.0
        )
        return jnp.where(ports.charge_sensitive, s_time, s_energy)

    def update_fairness_values(
        self, state: EnvState, ports: EVSE, scored: Array
    ) -> EnvState:
        """Add the customers marked in `scored` to the fairness bookkeeping."""
        s = self.customer_satisfaction(ports)
        scored = scored.astype(bool)
        fairness_cost = (scored * (1.0 - s) ** 2).sum()

        # Group 0 = time-sensitive, group 1 = charge-sensitive: shape (2, num_chargers)
        in_group = jnp.stack(
            [scored & ~ports.charge_sensitive, scored & ports.charge_sensitive]
        )
        g = in_group.astype(float)
        return state._replace(
            fairness_cost=state.fairness_cost + fairness_cost,
            group_served=state.group_served + g.sum(-1),
            group_s_sum=state.group_s_sum + (g * s).sum(-1),
            group_s_sq_sum=state.group_s_sq_sum + (g * s**2).sum(-1),
            group_unfair=state.group_unfair
            + (g * (s < self.fairness_threshold)).sum(-1),
            group_min_s=jnp.minimum(
                state.group_min_s, jnp.min(jnp.where(in_group, s, 1.0), axis=-1)
            ),
        )

    def score_cars_at_end_of_day(
        self, state: EnvState, ports: EVSE, is_last_step: Array
    ) -> EnvState:
        """Charge-sensitive cars only leave once they reach their target, so an agent
        could avoid the fairness cost by keeping them plugged in until the day ends.
        On the last step we therefore also score the charge-sensitive cars that are
        still connected, using the overtime they have built up so far. Time-sensitive
        cars leave at a fixed time the agent cannot change, so they are not affected."""
        still_connected = ports.charger_is_car_connected & ports.charge_sensitive
        return self.update_fairness_values(state, ports, still_connected & is_last_step)

    def add_new_cars(
        self, key: PRNGKeyArray, state: EnvState, ports: EVSE
    ) -> tuple[EnvState, EVSE]:
        key1, key2 = jax.random.split(key)

        new_cars_amount = self.get_num_cars_arriving(key1, state)

        # Generate new chargers and put the car_connected to False when:
        # 1. The index of the charger is already connected to a car
        # 2. There are less incoming cars than chargers
        not_connected_chargers = jnp.logical_not(ports.charger_is_car_connected)
        sort_order = jnp.argsort(not_connected_chargers, descending=True)
        required_chargers = jnp.arange(self.station.num_chargers) < new_cars_amount
        required_chargers_in_order = (
            jnp.zeros_like(required_chargers).at[sort_order].set(required_chargers)
        )
        arrival_of_new_car_positions = (
            required_chargers_in_order * not_connected_chargers
        )  # adjust for overflow
        incoming_chargers = self.get_new_cars_arriving(key2, state)
        incoming_chargers = incoming_chargers.replace(
            charger_is_car_connected=arrival_of_new_car_positions,
        )
        # Merge the incoming chargers with the current chargers
        merged_chargers = jax.tree.map(
            lambda new, curr: jax.lax.select(arrival_of_new_car_positions, new, curr),
            incoming_chargers,
            ports,
        )

        rejected_customers = jnp.maximum(
            new_cars_amount - not_connected_chargers.sum(), 0
        ).astype(jnp.int32)

        state = state._replace(
            rejected_customers=state.rejected_customers + rejected_customers
        )

        return state, merged_chargers

    def get_observation(self, state: EnvState) -> Array:

        observations = {
            "evses": state.grid.evses,
            "batteries": state.grid.batteries,
            "passives": state.grid.passives,
        }

        # Get future prices
        timesteps_per_hour = 60 // self.minutes_per_timestep
        hour_offsets = jnp.arange(self.price_hour_lookahead) * timesteps_per_hour
        future_timesteps = state.timestep + hour_offsets
        future_prices = jax.vmap(
            lambda t: self.get_grid_buy_price(state._replace(timestep=t))
        )(future_timesteps)
        future_sell_prices = jax.vmap(
            lambda t: self.get_grid_sell_price(state._replace(timestep=t))
        )(future_timesteps)

        # Calculate price differences for all lookahead periods
        price_diffs_buy = future_prices[1:] - future_prices[0]  # all diffs from now
        price_diffs_sell = future_sell_prices[1:] - future_sell_prices[0]  # ""

        observations.update(
            {
                "future_buy_prices": future_prices,
                "future_sell_prices": future_sell_prices,
                "future_price_diffs_buy": price_diffs_buy,
                "future_price_diffs_sell": price_diffs_sell,
                "current_timestep": state.timestep,
                "current_day_of_year": state.day_of_year,
                "is_workday": state.is_workday,
            }
        )

        return observations

    def get_reward(self, old_state: EnvState, new_state: EnvState) -> Array:
        profit_delta = new_state.profit - old_state.profit

        # uncharged_delta = new_state.uncharged_percentages - old_state.uncharged_percentages
        uncharged_delta = new_state.uncharged_kw - old_state.uncharged_kw
        charged_overtime_delta = new_state.charged_overtime - old_state.charged_overtime
        charged_undertime_delta = (
            new_state.charged_undertime - old_state.charged_undertime
        )
        rejected_customers_delta = (
            new_state.rejected_customers - old_state.rejected_customers
        )
        exceeded_capacity_delta = (
            new_state.exceeded_capacity - old_state.exceeded_capacity
        )
        battery_degredation_delta = (
            new_state.total_discharged_kw - old_state.total_discharged_kw
        )  # use discharged kw as proxy for degredation


        # New Normalized satisfaction reward values
        d_uncharged = new_state.sat_uncharged_norm - old_state.sat_uncharged_norm
        d_overtime = new_state.sat_overtime_norm - old_state.sat_overtime_norm
        d_undertime = new_state.sat_undertime_norm - old_state.sat_undertime_norm

        normalized_satisfaction_delta = (
            self.norm_w_uncharged * d_uncharged
            + self.norm_w_overtime * d_overtime
            + self.norm_w_undertime * d_undertime
        )

        return profit_delta - (
            self.charged_satisfaction_alpha * uncharged_delta
            + self.time_satisfaction_alpha
            * (charged_overtime_delta - (self.beta * charged_undertime_delta))
            + self.rejected_customers_alpha * rejected_customers_delta
            + self.capacity_exceeded_alpha * exceeded_capacity_delta
            + self.battery_degradation_alpha * battery_degredation_delta
            + self.norm_satisfaction_alpha * normalized_satisfaction_delta
            + self.fairness_alpha * (new_state.fairness_cost - old_state.fairness_cost)
        )

    def get_terminated(self, state: EnvState) -> bool:
        return False

    def get_truncated(self, state: EnvState) -> bool:
        return state.timestep >= self.max_episode_steps

    def get_info(
        self, state: EnvState, actions, old_state: EnvState = None
    ) -> Dict[str, Array]:
        return {
            "profit": state.profit,
            "exceeded_capacity": state.exceeded_capacity,
            "total_charged_kw": state.total_charged_kw,
            "total_discharged_kw": state.total_discharged_kw,
            "rejected_customers": state.rejected_customers,
            "served_customers": state.served_customers,
            "uncharged_percentages": state.uncharged_percentages,
            "uncharged_kw": state.uncharged_kw,
            "charged_overtime": state.charged_overtime,
            "charged_undertime": state.charged_undertime,
            "sat_uncharged_norm": state.sat_uncharged_norm,
            "sat_overtime_norm": state.sat_overtime_norm,
            "sat_undertime_norm": state.sat_undertime_norm,
            "fairness_cost": state.fairness_cost,
            **self.get_fairness_metrics(state),
        }

    def get_fairness_metrics(self, state: EnvState) -> Dict[str, Array]:
        """Three-layer fairness metrics. Group 0 = time-sensitive, 1 = charge-sensitive.
        A group without scored customers yet counts as fully satisfied (1.0)."""
        n, s_sum, sq_sum = state.group_served, state.group_s_sum, state.group_s_sq_sum
        has = n > 0
        # 1) Within-group fairness
        jain = jnp.where(has, s_sum**2 / jnp.maximum(n * sq_sum, 1e-8), 1.0)
        mean_s = jnp.where(has, s_sum / jnp.maximum(n, 1), 1.0)
        # Both groups pooled, kept for reference
        n_all, s_all, sq_all = n.sum(), s_sum.sum(), sq_sum.sum()
        jain_all = jnp.where(n_all > 0, s_all**2 / jnp.maximum(n_all * sq_all, 1e-8), 1.0)
        return {
            "jain_time": jain[0],
            "jain_charge": jain[1],
            "min_s_time": state.group_min_s[0],
            "min_s_charge": state.group_min_s[1],
            "unfair_time": state.group_unfair[0],
            "unfair_charge": state.group_unfair[1],
            "mean_s_time": mean_s[0],
            "mean_s_charge": mean_s[1],
            # 2) Between-group fairness: gap in mean satisfaction
            "group_gap": jnp.abs(mean_s[0] - mean_s[1]),
            # 3) Summary: the worse of the two groups
            "worst_group_jain": jnp.min(jain),
            "worst_group_min_s": jnp.min(state.group_min_s),
            "jain_overall": jain_all,
            "mean_s_overall": jnp.where(n_all > 0, s_all / jnp.maximum(n_all, 1), 1.0),
        }

    def kw_to_kw_this_timestep(self, kw_drawn: Float[Array, "..."]) -> Array:
        return kw_drawn / 60 * self.minutes_per_timestep

    @property
    def observation_space(self):
        obs, _ = self.reset_env(jax.random.PRNGKey(0))
        return jax.tree.map(
            lambda v: jym.Box(-jnp.inf, jnp.inf, getattr(v, "shape", ())), obs
        )

    @property
    def action_space(self) -> jym.Space:
        """
        Define the action space of the environment.
        """
        num_actions_per_charger = self.num_discretization_levels
        if self.allow_discharging:
            num_actions_per_charger *= 2
        num_actions_per_battery = self.num_discretization_levels * 2
        num_actions_per_charger += 1  # idle action
        num_actions_per_battery += 1  # idle action

        actions = {
            "evses": jax.tree.map(
                lambda item: jym.MultiDiscrete(
                    np.full(item.num_chargers, num_actions_per_charger)
                ),
                self.station.evses,
                is_leaf=lambda x: isinstance(x, EVSE),
            ),
            "batteries": jax.tree.map(
                lambda item: jym.Discrete(num_actions_per_battery),
                self.station.batteries,
                is_leaf=lambda x: isinstance(x, StationBattery),
            ),
        }
        return actions