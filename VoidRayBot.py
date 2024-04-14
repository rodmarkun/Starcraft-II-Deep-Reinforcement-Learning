# General imports
import random
import constants
import numpy as np
from sc2.ids.unit_typeid import UnitTypeId

# SC2 API imports
from sc2.bot_ai import BotAI  # parent class we inherit from
from sc2.ids.unit_typeid import UnitTypeId

class VRBot(BotAI): # inhereits from BotAI (part of BurnySC2)
    def __init__(self, *args,bot_in_box=None, action_in=None, result_out=None, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.action_in = action_in
        self.result_out = result_out
        self.main_base_destroyed = False

    async def on_end(self,game_result):
        print ("Game over!")
        
        obs = constants.EMPTY_OBSERVATION

        print(f"GAME DURATION: {self.time}")
        
        if str(game_result) == "Result.Victory":
            reward = int(500 + 1800 / self.time * 100) # The faster the victory, the better
        else:
            reward = 0

        self.result_out.put({"observation" : obs, "reward" : reward, "action" : None, "done" : True, "truncated" : False, "info" : {}})
        

    async def on_step(self, iteration): # on_step is a method that is called every step of the game.
        game_time_minutes = self.time / 60  # self.time is in seconds

        if game_time_minutes > 30:
            print("30 minutes have passed. Ending the game...")
            # Force a game end. Here, we just surrender.
            await self.client.leave()


        self.action = self.action_in.get()

        if iteration % 10 == 0:
            print(f"Iteration: {iteration} | Action: {self.action}")

        if self.action is None:
            print("no action returning.")
            return None
        
        # DRL Agent Logic
        # 0 - Expand, Build Assimilators and Workers (Long-Term Economy)
        # 1 - Build Stargate (Military Building)
        # 2 - Build Void Rays (Military Unit)
        # 3 - Attack with all Void Rays (Attacking the enemy to win rewards)
        # 4 - Build Pylon (Supply)
        # 5 - Do nothing 

        try:
            await self.distribute_workers()
            if self.action == 0:
                await self.expand()
                await self.build_assimilators()
                await self.build_workers()
            elif self.action == 1:
                await self.build_gateway()
                await self.build_cybernetics_core()
                await self.build_stargates()
            elif self.action == 2:
                await self.build_void_rays()
            elif self.action == 3:
                await self.attack()
            elif self.action == 4:
                await self.build_pylons()
            else:
                pass
        except Exception as e:
            print(f"Exception - {e}")

        obs = self.visualize_intel()
        reward = self.reward_function()
        if iteration % 10 == 0:
            print(f"Reward: {reward}")

        self.result_out.put({"observation" : obs, "reward" : reward, "action" : None, "done" : False, "truncated" : False, "info" : {}})

    async def build_workers(self):
        for nexus in self.structures(UnitTypeId.NEXUS).ready:
            if nexus.is_idle and self.can_afford(UnitTypeId.PROBE):
                nexus.train(UnitTypeId.PROBE)

    async def build_pylons(self):
        if not self.already_pending(UnitTypeId.PYLON):
            if self.supply_left < 5:
                if len(self.structures(UnitTypeId.NEXUS)) > 0:
                    nexus = self.structures(UnitTypeId.NEXUS).ready.random
                    position = nexus.position.towards(self.game_info.map_center, 5)
                    if self.can_afford(UnitTypeId.PYLON):
                        await self.build(UnitTypeId.PYLON, near=position)

    async def build_assimilators(self):
        if len(self.structures(UnitTypeId.NEXUS)) > 0:
            for nexus in self.structures(UnitTypeId.NEXUS).ready:
                vespenes = self.vespene_geyser.closer_than(15.0, nexus.position)  # Corrected to use `position`
                for vespene in vespenes:
                    if not self.can_afford(UnitTypeId.ASSIMILATOR) or self.structures(UnitTypeId.ASSIMILATOR).closer_than(1.0, vespene).exists:
                        continue
                    worker = self.select_build_worker(vespene.position)
                    if worker:
                        worker.build(UnitTypeId.ASSIMILATOR, vespene)

    async def expand(self):
        if self.can_afford(UnitTypeId.NEXUS) and not self.already_pending(UnitTypeId.NEXUS):
            location = await self.get_next_expansion()  # Assuming this is async
            if location:
                await self.build(UnitTypeId.NEXUS, near=location)

    async def build_gateway(self):
        if len(self.structures(UnitTypeId.NEXUS)) > 0:
            if not self.structures(UnitTypeId.GATEWAY).ready.exists and not self.already_pending(UnitTypeId.GATEWAY):
                pylons = self.structures(UnitTypeId.PYLON).ready
                if pylons.exists and self.can_afford(UnitTypeId.GATEWAY):
                    await self.build(UnitTypeId.GATEWAY, near=pylons.closest_to(self.structures(UnitTypeId.NEXUS).first))

    async def build_cybernetics_core(self):
        if len(self.structures(UnitTypeId.NEXUS)) > 0:
            if self.structures(UnitTypeId.GATEWAY).ready.exists and not self.structures(UnitTypeId.CYBERNETICSCORE):
                pylon = self.structures(UnitTypeId.PYLON).ready.random
                if self.can_afford(UnitTypeId.CYBERNETICSCORE) and not self.already_pending(UnitTypeId.CYBERNETICSCORE):
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon.position.towards(self.game_info.map_center, 5))

    async def build_stargates(self):
        if len(self.structures(UnitTypeId.NEXUS)) > 0:
            if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                for pylon in self.structures(UnitTypeId.PYLON).ready:
                    if self.can_afford(UnitTypeId.STARGATE) and not self.already_pending(UnitTypeId.STARGATE):
                        await self.build(UnitTypeId.STARGATE, near=pylon.position.towards(self.game_info.map_center, 5))

    async def build_void_rays(self):
        for stargate in self.structures(UnitTypeId.STARGATE).ready.idle:
            if self.can_afford(UnitTypeId.VOIDRAY) and self.supply_left > 0:
                stargate.train(UnitTypeId.VOIDRAY)

    async def defend_bases(self):
        if self.units(UnitTypeId.VOIDRAY).amount < 5:
            for vr in self.units(UnitTypeId.VOIDRAY).idle:
                vr.move(self.start_location)

    async def attack(self):
        # take all void rays and attack!
        for voidray in self.units(UnitTypeId.VOIDRAY).idle:
            if self.enemy_units.closer_than(10, voidray):
                voidray.attack(random.choice(self.enemy_units.closer_than(10, voidray)))
            elif self.enemy_units:
                voidray.attack(random.choice(self.enemy_units))
            elif self.enemy_structures.closer_than(10, voidray):
                voidray.attack(random.choice(self.enemy_structures.closer_than(10, voidray)))
            elif self.enemy_structures:
                voidray.attack(random.choice(self.enemy_structures))
            elif self.enemy_start_locations:
                voidray.attack(self.enemy_start_locations[0])
        
    def reward_function(self):
        reward = 0
        attack_count = 0
        # iterate through our void rays:
        for vr in self.units(UnitTypeId.VOIDRAY):
            # if voidray is attacking and is in range of enemy unit:
            if vr.is_attacking and vr.target_in_range:
                if self.enemy_units.closer_than(12, vr) or self.enemy_structures.closer_than(12, vr):
                    reward += 0.02  
                    attack_count += 1

        return reward
    
    def visualize_intel(self):

        obs = constants.EMPTY_OBSERVATION
        limits = np.array(constants.OBSERVATION_SPACE_ARRAY)

        # Number of probes
        obs[0] = self.units(UnitTypeId.PROBE).amount
        # Number of void rays
        obs[1] = self.units(UnitTypeId.VOIDRAY).amount
        # Number of attacking void rays
        obs[2] = len([vr for vr in self.units(UnitTypeId.VOIDRAY) if vr.is_attacking])
        # Number of nexus
        obs[3] = self.units(UnitTypeId.NEXUS).amount
        # Number of assimilators
        obs[4] = self.units(UnitTypeId.ASSIMILATOR).amount
        # Number of stargates
        obs[5] = self.units(UnitTypeId.STARGATE).amount
        # Number of pylons
        obs[6] = self.units(UnitTypeId.PYLON).amount
        # Supply left
        obs[7] = min(199, int(self.supply_left))
        # Game time
        obs[8] = self.time

        for i in range(len(obs)):
            if obs[i] >= limits[i]-1:
                obs[i] = limits[i] - 1
        return obs