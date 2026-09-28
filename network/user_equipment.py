from utils import *
from config import *
from . import config
from .channel import compute_channel_gain
from traffic.config import packetsPerUE

class UEStatus(enum.IntEnum):
    IDLE = 0
    WAITING = 1
    ACTIVE = 2
    DONE = 3
    DROPPED = 4


class UserEquipment:
    height: float = config.ueHeight
    signal_thresh = config.signalThreshold
    tau_c = config.tau_c
    tau_p = config.tau_p
    record_sinr = True
    _cache = defaultdict(dict)

    def __init__(self, pos, service, demand, delay_budget):
        self.id = id(self)
        self.pos = np.asarray(pos)
        self.bss = []
        self.pilot = None
        self.net = None
        self.service = service
        self.status = UEStatus.IDLE
        self.total_demand = demand  # full session commitment
        self.packet_size = demand / packetsPerUE
        self.demand = self.packet_size  # current packet only
        self.packets_remaining = packetsPerUE  # includes current packet
        self.completed_packets = 0
        self.dropped_packets = 0
        self.dropped_bits = 0.
        self.packet_delay_sum = 0.
        self.delay_budget = delay_budget
        self.delay = 0.
        self.t_served = 0.
        self._dists = None
        self._gains = None
        self._gamma = None
        self._delta = None
        self._thruput = None
        self._cover_bss = set()
        self.cluster_size = 0
        self._sinr_stats = Counter()
        self.connection_requested = False
        self.prelog_factor = (self.tau_c - self.tau_p) / self.tau_c

    # define boolean properties for each UE status
    for status in UEStatus._member_names_:
        exec(f"""@property\ndef {status.lower()}(self):
             return self.status == UEStatus.{status}""")

    def cached_property(func, _C=_cache):
        @property
        @wraps(func)
        def wrapped(self):
            cache = _C[self.id]
            if self.delay != cache.get('_t'):
                cache.clear()
                cache['_t'] = self.delay
            else:
                return cache[None]
            ret = func(self)
            cache[None] = ret
            return ret
        return wrapped

    @property
    def distances(self):
        return self._dists
    
    @distances.setter
    def distances(self, dists):
        self._dists = dists
        return dists

    @property
    def channel_gains(self):
        return self._gains
    
    @channel_gains.setter
    def channel_gains(self, gains):
        self._gains = gains
        self._gamma = self._gains
        self._delta = np.zeros(self.net.num_bs)
        self._sorted_bss = np.argsort(self._gains)[::-1]
    
        # self._cover_bss.update(self.net.get_bs(i) for i in self._sorted_bss[:self.cluster_size])
        total_gain = np.sum(gains)
        cumulative_gain = 0
        self._cover_bss.clear()
        for i in self._sorted_bss:
            cumulative_gain += self._gains[i]
            self._cover_bss.add(self.net.get_bs(i))
            if cumulative_gain >= 0.9 * total_gain:
                break
        self.cluster_size = len(self._cover_bss)
        for b in self._cover_bss:
            b.add_to_cell(self)
        return gains

    @property
    def channel_gain(self):
        return [self.channel_gains[b.id] for b in self.bss]

    @property
    def tx_power(self):
        if self.active:
            p = 0
            for b in self.bss:
                if self in b.ues.values():
                    p += b.power_alloc[self.id]
            return p
        return 0.
    
    @property
    def signal_power(self):
        if not self.active:
            return 0.
        S = 0
        for b in self.bss:
            if self in b.ues.values():
                S += np.sqrt((b.num_ant - b.tau_sl) * b.power_alloc[self.id] * self._gamma[b.id])
        return S ** 2
    
    @property
    def interference(self):
        if not self.active: return 0.
        I1 = 0
        for u in self.net.pilot_usage[self.pilot]:
            if u != self:
                for b in u.bss:
                    if u in b.ues.values():
                        I1 += np.sqrt((b.num_ant - b.tau_sl) * b.power_alloc[u.id] * self._gamma[b.id])
        I2 = 0
        for u in self.net.ues.values():
            for b in u.bss:
                if u in b.ues.values():
                    I2 += b.power_alloc[u.id] * (self._gains[b.id] - self._delta[b.id] * self._gamma[b.id])    
        return I1 ** 2 + I2
    
    @property
    def data_rate(self):
        if self._thruput is None:
            self._thruput = self.compute_data_rate()
            if DEBUG:
                debug('UE {}: computed data rate {:.2f} mb/s'.format(self.id, self._thruput/1e6))
        return self._thruput
    
    @property
    def time_limit(self):
        return self.delay_budget - self.delay
    
    @property
    def required_rate(self):
        t_lim = self.time_limit
        if t_lim <= 0: return 0.
        return self.demand / t_lim
    
    @property
    def throughput_ratio(self):
        if self.required_rate <= 0: return 1.
        return min(self.data_rate / self.required_rate, 10.)

    def compute_sinr(self, N=config.noiseSpectralDensity):
        if not self.bss: return 0
        self._S = self.signal_power
        self._I = self.interference
        self._SINR = self._S / (self._I + N * self.bss[0].bandwidth)
        if self.record_sinr:
            self._sinr_stats.update(
                T = 1,
                tx_power = self.tx_power,
                chan_gain = self.channel_gain,
                signal = self._S,
                interference = self._I,
                sinr = self._SINR
            )
        return self._SINR

    @property
    def SE(self):
        sinr = self.compute_sinr()
        return (self.tau_c - self.tau_p) / self.tau_c * np.log2(1 + sinr)

    @timeit
    def compute_data_rate(self):
        """
        Returns:
        The data_rate of the UE in bits/s.
        """
        sinr = self.compute_sinr()
        if sinr == 0: return 0
        return self.prelog_factor * self.bss[0].bandwidth * np.log2(1 + sinr)

    def update_data_rate(self):
        self._thruput = None
        if self.active:
            for b in self.bss:
                b.update_power_consumption()

    def update_status(self):
        if self.bss:
            for b in self.bss:
                if self in b.ues.values():
                    self.status = UEStatus.ACTIVE
                    break
            else:
                self.status = UEStatus.WAITING
        else:
            self.status = UEStatus.IDLE

    # def add_to_queue(self, bs):
    #     if self._in_queue_of_bs is bs: return
    #     self.remove_from_queue()
    #     self._in_queue_of_bs = bs
    #     bs.queue[self.id] = self
    #     bs.schedule_action()
    #     info(f'UE {self.id}: added to queue of BS {bs.id}')

    # def remove_from_queue(self):
    #     if self._in_queue_of_bs:
    #         bs = self._in_queue_of_bs
    #         bs.queue.pop(self.id)
    #         self._in_queue_of_bs = None
    #         info(f'UE {self.id}: removed from queue of BS {bs.id}')

    def request_connection(self):
        self.connection_requested = True
        for b in self._cover_bss:
            if b not in self.bss:
                res = b.respond_connection_request(self)
        # for i in self._sorted_bss:
        #     bs = self.net.get_bs(i)
        #     if bs not in self.bss:
        #         res = bs.respond_connection_request(self)
        #         if res:
        #             self._cover_bss.add(bs)
        #             bs.add_to_cell(self)
        #             if len(self.bss) >= self.cluster_size:
        #                 break
        return res

    # def request_connection(self):
    #     assert len(self.bss) < self.cluster_size
    
    #     res = False                                          
    #     for i in self._sorted_bss[:self.cluster_size]:        # **只看前 C 个**
    #         bs = self.net.get_bs(i)
    #         if bs in self.bss:                                # 已连接就跳过
    #             continue
    #         if bs.respond_connection_request(self):
    #             bs.add_to_cell(self)
    #             res = True
    #             if len(self.bss) >= self.cluster_size:        # 已够用，停止
    #                 break
    #     return res

    def disconnect(self, bs=None):
        if bs is not None:
            bs._disconnect(self.id)
        else:
            if self.bss is None: return
            for b in list(self.bss):
                if self in b.ues.values():
                    b._disconnect(self.id)
                else:
                    b.pop_from_queue(self)
        self.update_data_rate()

    @property
    def future_packets(self):
        return max(0, self.packets_remaining - 1)

    def finish_packet(self, success):
        """Record immediately, keeping the UE association until the final packet."""
        result = 0 if success else 1
        for owner in (self.net, *self._cover_bss):
            owner._packet_stats[self.service, result] += 1
            owner._packet_totals[self.service, result] += 1
        self.packet_delay_sum += self.delay
        if success:
            self.completed_packets += 1
        else:
            self.dropped_packets += 1
            self.dropped_bits += self.demand
        self.packets_remaining -= 1
        self.__class__._cache.pop(self.id, None)
        if not self.packets_remaining:
            self.quit()
        else:
            self.demand = self.packet_size
            self.delay = 0.

    def quit(self):
        self.disconnect()
        service = self.service
        if self.dropped_packets:
            remaining_ratio = self.dropped_bits / self.total_demand
            for b in self._cover_bss:
                b._ue_stats[service, 1] += [1, remaining_ratio]
        else:
            for b in self._cover_bss:
                b._ue_stats[service, 0] += [1, self.packet_delay_sum / (packetsPerUE * self.delay_budget)]
        for b in list(self._cover_bss):
            b.remove_from_cell(self)
            self._cover_bss.remove(b)
        if not self.dropped_packets:
            self.demand = 0.
            self.status = UEStatus.DONE
        else:
            self.status = UEStatus.DROPPED
        # self.served = self.total_demand - self.demand
        if EVAL:
            dropped = self.dropped_bits
            done = self.total_demand - dropped
            delay = self.packet_delay_sum / (self.completed_packets + self.dropped_packets)
            service_time = self.t_served
            steps = self._sinr_stats.pop('T', 1)
            self.net.add_stat('ue_stats', dict(
                demand = self.total_demand,
                done = done,
                dropped = dropped,
                delay = delay,
                delay_budget = self.delay_budget,
                service_time = service_time,
                completed_packets = self.completed_packets,
                dropped_packets = self.dropped_packets,
                # **{'avg_'+k: self._sinr_stats[k]/steps
                #    for k in list(self._sinr_stats)}
            ))
        self.net.remove_user(self.id)
        self.__class__._cache.pop(self.id, None)
    
    def step(self, dt):
        # Resolve at most one packet per step; discard excess capacity.
        if not self.packets_remaining:
            return
        rate = max(0., self.data_rate) if self.active else 0.
        self.demand = max(0., self.demand - rate * dt)
        self.delay += dt
        if self.active:
            self.t_served += dt
        # Boundary-only checks: completion takes precedence. A new packet is
        # prepared at this boundary but first served in the following step.
        if self.demand <= 0:
            self.finish_packet(True)
        elif self.delay >= self.delay_budget:
            self.finish_packet(False)

    @timeit
    def info_dict(self):
        if self.bss:
            bs_id = [b.id for b in self.bss]
        else:
            bs_id = ['-']
        return dict(
            bs_id=bs_id,
            status=self.status.name,
            demand=self.demand / 1e3,   # in kb
            rate=self.compute_data_rate() / 1e6,  # in mb/s
            ddl=self.time_limit * 1e3,  # in ms
            future_packets=self.future_packets,
        )
    
    def __repr__(self):
        return 'UE(%d)' % self.id
        return 'UE({})'.format(kwds_str(
            id=self.id,
            # pos=self.pos,
            **self.info_dict()
        ))


class TestUE(UserEquipment):
    record_sinr = False
    
    def __init__(self, net, grid_size=config.probeGridSize):
        super().__init__(None, None, 0, 999)
        self.net = net
        self.x = np.linspace(0, net.area[0], round(net.area[0] / grid_size) + 1)
        self.y = np.linspace(0, net.area[1], round(net.area[1] / grid_size) + 1)
        
    @property
    def tx_power(self):
        return self._tx_power

    def test_sinr(self):
        """ Measure SINR over the grid. """
        print('Probing SINR...')
        self.status = UEStatus.ACTIVE
        # bss = np.arange(self.net.num_bs)
        poses = np.array(list(itertools.product(self.x, self.y, [self.height])))
        csi_keys = ['S', 'I', 'SINR'] # + [f'C_{i}' for i in range(self.net.num_bs)]
        # csi_index = pd.MultiIndex.from_product([self.x, self.y, bss], names=['x', 'y', 'bs'])
        csi_index = pd.MultiIndex.from_product([self.x, self.y], names=['x', 'y'])
        csi = np.zeros((len(csi_index), len(csi_keys)))
        distances = np.sqrt(np.sum((poses[:,None] - self.net.bs_positions)**2, axis=-1))
        channel_gains = compute_channel_gain(distances)
        N = config.noiseSpectralDensity
        i = 0
        for dists, gains in zip(distances, channel_gains):
            self.distances = dists
            self.channel_gains = gains
            for c in self._cover_cells:
                bs = self.net.get_bs(c)
                if not self.bs and bs.responding:
                    self.bs = bs
                    # G = self.channel_gain
                    if bs.sleep or bs.ues_full:
                        S = I = 0
                    else:
                        self._tx_power = bs.transmit_power / (bs.num_ue + 1)
                        S = self.signal_power
                        I = self.interference
                    SINR = S / (I + N)
                bs.remove_from_cell(self)
            if not self.bs:
                S = I = SINR = 0
            self.bs = None
            cover_cells = [0] * self.net.num_bs
            for c in self._cover_cells:
                cover_cells[c] = 1
            csi[i] = list(map(lin2dB, [S, I, SINR]))
            i += 1
        return pd.DataFrame(csi, index=csi_index, columns=csi_keys)
