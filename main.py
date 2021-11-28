import queue
import inspect

import numpy
from functools import reduce


class Generator:
    p = None
    k = None
    b = None

    _instance = None
    initialized = False

    def __init__(self, k: int, b: int, p: int = 13):
        self.k = k
        self.b = b
        self.p = p

    @classmethod
    def get_instance(cls) -> 'Generator':
        return cls._instance

    @classmethod
    def initialize(cls, k: int, b: int, p: int = 13):
        cls._instance = Generator(k, b, p)
        cls.initialized = True

    def next(self, a: int = None, b: int = None):
        if a is None:
            a = 0
        else:
            a = int(a)
        if b is None:
            b = self.b
        else:
            b = int(b)
        self.p = a + (self.p * self.k) % (b - a)
        return self.p / (b - a)

    def uniform(self, a: float, b: float):
        return a + (b - a) * self.next()

    def normal(self, m: float, sigma: float, n: int = 6):
        n = int(n)
        _sum = sum([self.next() for _ in range(n)])
        return m + sigma * int(numpy.sqrt((12 + n - 1) / n)) * (_sum - int(n / 2))

    def exponential(self, alpha: float):
        return -(1 / alpha) * numpy.log(self.next())

    def gamma(self, alpha: float, nu: int):
        return -(1 / alpha) * numpy.log(reduce(lambda x, y: x * y, [self.next() for _ in range(nu)], 1))

    def triangle(self, a: float, b: float, inv: bool = False):
        _func = numpy.amax if inv else numpy.amin
        return a + (b - a) * _func([self.next(), self.next()])

    def simpson(self, a: float, b: float):
        args = [self.uniform(a / 2, b / 2) for _ in range(2)]
        return sum(args)


class QueueSystem:
    class Token:
        pass

    class TerminationStub:
        pass

    class NodeBase:
        _sources = None
        _consumers = None

        _tick_counter = None
        _busy_counter = None

        # limit consume() call to prevent loops in graph
        _consume_counter = None

        # limit tick() call to prevent loops in graph
        _tick_call_counter = None

        def __init__(self, src=None):
            self._sources = []
            if isinstance(src, QueueSystem.NodeBase):
                self._sources.append(src)
                src.subscribe(self)
            elif type(src) is list:
                for item in src:
                    self._sources.append(item)
                    item.subscribe(self)
            self._consumers = []
            self._tick_counter = 0
            self._busy_counter = 0
            self._consume_counter = 0
            self._tick_call_counter = 0

        def subscribe(self, dst=None):
            if type(self._consumers) is list:
                if isinstance(dst, QueueSystem.NodeBase):
                    self._consumers.append(dst)

        def terminate(self):
            self._consumers = tuple([QueueSystem.TerminationStub()])

        def consume(self):
            if self._consume_counter >= len(self._consumers):  # <= 0:
                raise Exception("Too many consume call for node")
            self._consume_counter += 1  # -=1

        def tick(self):
            if self._tick_call_counter == 0:
                self.run()
                for parent in self._sources:
                    parent.tick()
                self._consume_counter = 0  # len(self._consumers)
                self._tick_counter += 1
            self._tick_call_counter += 1
            if self._tick_call_counter == max(len(self._consumers), 1):
                self._tick_call_counter = 0

        def run(self):
            raise NotImplementedError

        def get_busy_rate(self):
            return self._busy_counter / self._tick_counter

        def get_tick_counter(self):
            return self._tick_counter

        def get_busy_counter(self):
            return self._busy_counter

    class Source(NodeBase):
        _static = None
        _blocking = None
        _gen_prob = None
        _gen_rate = None
        _gen_counter = None
        _blocked = None

        _generated_value = None
        _generated_counter = None
        _discarded_counter = None

        def __init__(self, static: bool = False, blocking: bool = False, **kwargs):
            super().__init__(None)
            self._static = static
            if static:
                self._gen_rate = kwargs.get('gen_rate')
                if type(self._gen_rate) is not int:
                    raise ValueError
                self._gen_counter = 0
            else:
                self._gen_prob = kwargs.get('gen_prob')
            self._blocking = blocking
            self._discarded_counter = 0
            self._generated_counter = 0
            self._blocked = False

        def run(self):
            if self._generated_value is not None:
                if self._blocking:
                    self._blocked = True
                else:
                    self._discarded_counter += 1
                    self._generated_value = None
            if not self._blocked:
                self._busy_counter += 1
                if self._static:
                    self._gen_counter += 1
                    if self._gen_counter >= self._gen_rate:
                        self._generated_value = QueueSystem.Token()
                        self._generated_counter += 1
                        self._gen_counter = 0
                else:
                    if Generator.get_instance().next() <= self._gen_prob:
                        self._generated_value = QueueSystem.Token()
                        self._generated_counter += 1

        def consume(self):
            super().consume()
            ret_value = self._generated_value
            self._generated_value = None
            if self._blocking:
                self._blocked = False
            return ret_value

        def get_discard_rate(self):
            return self._discarded_counter / self._generated_counter

        def get_generation_rate(self):
            return self._generated_counter / self._busy_counter

        def get_blocked_rate(self):
            return 1 - self.get_busy_rate()

        def get_generated_counter(self):
            return self._generated_counter

        def get_discarded_counter(self):
            return self._discarded_counter

        def get_blocked_counter(self):
            return self._tick_counter - self._busy_counter

    class Queue(NodeBase):
        _capacity = None

        _queue = None
        _served_counter = None
        _overall_counter = None
        _full_counter = None

        def __init__(self, src, capacity: int = 2):
            super().__init__(src)
            self._capacity = capacity
            self._served_counter = 0
            self._overall_counter = 0
            self._full_counter = 0
            self._queue = queue.Queue(capacity)

        def run(self):
            for parent in self._sources:
                if not self._queue.full():
                    temp = parent.consume()
                    if temp is not None:
                        self._queue.put(temp, False)
                        self._served_counter += 1
                else:
                    break

            self._overall_counter += self._queue.qsize()
            if not self._queue.empty():
                self._busy_counter += 1
                if self._queue.full():
                    self._full_counter += 1

        def consume(self):
            super().consume()
            if not self._queue.empty():
                return self._queue.get(False)
            return None

        def get_average_load(self):
            return self._overall_counter / self._busy_counter

        def get_full_rate(self):
            return self._full_counter / self._busy_counter

        def get_average_wait(self):
            return self._overall_counter / self._served_counter

        def get_served_counter(self):
            return self._served_counter

        def get_full_counter(self):
            return self._full_counter

    class Server(NodeBase):
        _blocking = None
        _static = None
        _delay_counter = None
        _serve_prob = None
        _blocked = None

        _served_value = None
        _service_finished = None

        _blocked_counter = None
        _served_counter = None
        _discarded_counter = None
        _delayed_counter = None

        _delay_callback = None

        def __init__(self, src, static: bool = False, blocking: bool = False, **kwargs):
            super().__init__(src)
            self._static = static
            if static:
                self._delay_callback = kwargs.get('callback')
                # if type(self._delay_callback) is not callable:
                #     raise ValueError
                self._delay_counter = 0
            else:
                self._serve_prob = kwargs.get('serve_prob')
            self._blocking = blocking
            self._blocked = False
            self._blocked_counter = 0
            self._discarded_counter = 0
            self._delayed_counter = 0
            self._served_counter = 0
            self._service_finished = False

        def run(self):
            if self._service_finished:
                if self._served_value is not None:
                    if self._blocking:
                        self._blocked = True
                    else:
                        self._discarded_counter += 1
                        self._served_value = None
                        self._service_finished = False
            if not self._blocked:
                if self._served_value is None:
                    for parent in self._sources:
                        temp = parent.consume()
                        if temp is not None:
                            self._served_value = temp
                            self._service_finished = False
                            if self._static:
                                self._delay_counter = int(self._delay_callback()) + 1
                                # self._delay_counter = round(self._delay_callback())
                            break
                if self._served_value is not None:
                    if self._static:
                        self._delay_counter -= 1
                        if self._delay_counter <= 0:
                            self._served_counter += 1
                            self._service_finished = True
                        else:
                            self._delayed_counter += 1
                    else:
                        if Generator.get_instance().next() <= self._serve_prob:
                            self._served_counter += 1
                            self._service_finished = True
                        else:
                            self._delayed_counter += 1
                    self._busy_counter += 1
            else:
                self._blocked_counter += 1

        def consume(self):
            super().consume()
            ret_value = None
            if self._service_finished:
                ret_value = self._served_value
                self._served_value = None
                self._service_finished = False
                if self._blocking:
                    self._blocked = False
            return ret_value

        def get_serve_rate(self):
            return self._served_counter / self._busy_counter

        def get_discard_rate(self):
            return self._discarded_counter / self._busy_counter

        def get_blocked_rate(self):
            return self._blocked_counter / self._tick_counter

        def get_blocked_counter(self):
            return self._blocked_counter

        def get_served_counter(self):
            return self._served_counter

        def get_discarded_counter(self):
            return self._discarded_counter

        pass

    pass


def lab():
    Generator.initialize(102191, 203563, 131)

    params = [
        2,
        0,
        2.5,
        3
    ]
    _options = ["uniform", "normal", "exponential", "gamma", "triangle"]    # , "simpson"
    _labels = [
        f"Input flow rate: [{params[0]}] ",
        f"Queue capacity: [{params[1]}] ",
        f"Output flow rate: [{params[2]}] ",
        f"".join(f"\n{i+1}. {_options[i]}" for i in range(len(_options))) + f"\nProbability distribution: [{params[3]}] "
    ]
    _types = [
        float,
        int,
        float,
        int
    ]
    _validators = [
        lambda x: 2 if x < 0 else x,
        lambda x: x,
        lambda x: 2.5 if x < 0 else x,
        lambda x: x-1 if x-1 in range(len(_options)) else 0,
    ]
    for i in range(len(params)):
        try:
            temp = _types[i](input(_labels[i]))
            params[i] = temp if temp != 0 else params[i]
        except ValueError:
            pass
        params[i] = _validators[i](params[i])

    _option_labels = [
        f"1/(b-a)",
        f"(1/(sigma * sqrt(2*pi)) * e^(-(x-m)/(2*sigma^2))",
        f"alpha * e^(-alpha * x)",
        f"((alpha^nu)/((nu - 1)!)) * x^(nu-1) * e^(-alpha * x)",
        f"2*(x-a)/((b-a)^2)",
        f"4*[(x-a)|(b-x)]/((b-a)^2)",
    ]

    _target_gen_prob = 0.8
    _scale_rate = 5
    params[2] = params[2] / (params[0] / _target_gen_prob) / _scale_rate
    params[0] = _target_gen_prob / _scale_rate

    _generator_params = [
        [0.0, 2*(1/params[2])],
        [1/params[2], (1/params[2])/3, 6],
        [params[2]],
        [params[2] / 3, 3],
        [0, (3*(1/params[2]))**0.5, False]
    ]
    print(_option_labels[params[3]])

    _callables = [
        Generator.get_instance().uniform,
        Generator.get_instance().normal,
        Generator.get_instance().exponential,
        Generator.get_instance().gamma,
        Generator.get_instance().triangle,
        Generator.get_instance().simpson
    ]

    _call_args = dict()
    _callable_arg_names = inspect.getfullargspec(_callables[params[3]])[0]
    _callable_arg_types = getattr(_callables[params[3]], "__annotations__")
    for i in range(1, len(_callable_arg_names)):
        _call_args[_callable_arg_names[i]] = \
            _callable_arg_types[_callable_arg_names[i]](_generator_params[params[3]][i-1])

    def _call():
        __call_temp = _callables[params[3]](**_call_args)
        return __call_temp if __call_temp > 0 else 0

    nodes = []
    nodes.append(QueueSystem.Source(False, True, gen_prob=params[0]))
    nodes.append(QueueSystem.Queue(nodes[0], params[1]))
    nodes.append(QueueSystem.Server(nodes[1], True, False, callback=_call))

    terminals = [nodes[-1]]
    for item in terminals:
        item.terminate()

    for _ in range(int(1e6)):
        for item in terminals:
            item.tick()
            item.consume()

    node_codes = [0] * 3
    print(f"\n\tQueue system statistics: ")
    for node in nodes:
        if type(node) is QueueSystem.Source:
            node_codes[0] += 1
            print(f"S{node_codes[0]} statistics: ")
            print(f"\tBusy rate: {node.get_busy_rate()}; "
                  f"\tgeneration rate: {node.get_generation_rate()}; "
                  f"\tdiscard rate: {node.get_discard_rate()}; "
                  f"\tblocked rate: {node.get_blocked_rate()}")
        elif type(node) is QueueSystem.Queue:
            node_codes[1] += 1
            print(f"Q{node_codes[1]} statistics: ")
            print(
                f"\tBusy rate: {node.get_busy_rate()}; " 
                f"\taverage wait duration: {node.get_average_wait()}; "
                f"\taverage load rate: {node.get_average_load()}; "
                f"\n\tfull queue rate: {node.get_full_rate()}")
        elif type(node) is QueueSystem.Server:
            node_codes[2] += 1
            print(f"C{node_codes[2]} statistics: ")
            print(f"\tBusy rate: {node.get_busy_rate()}; " 
                  f"\tserve rate: {node.get_serve_rate()}; "
                  f"\tdiscard rate: {node.get_discard_rate()}; "
                  f"\tnode blocked rate: {node.get_blocked_rate()}")

    print(f"Total queue system clock count: {nodes[-1].get_tick_counter()} / {nodes[0].get_tick_counter()} \n")

    total_discarded = 0
    total_blocked_clocks = 0
    total_tokens_generated = 0
    total_tokens_served = 0
    average_block_rate = 0.0
    average_queue_len = 0.0
    average_queue_wait = 0.0
    average_serve_duration = 0.0
    average_busy_rate = 0.0
    total_blocked_nodes = 0
    total_queue_nodes = 0
    total_token_load = 0.0

    for node in nodes:
        blocked = False
        if type(node) is QueueSystem.Source:
            total_discarded += node.get_discarded_counter()
            total_tokens_generated += node.get_generated_counter()
            total_token_load += node.get_blocked_rate()
            average_serve_duration += \
                (node.get_generated_counter() + node.get_blocked_counter()) / node.get_generated_counter()

            _blocked_counter = node.get_blocked_counter()
            total_blocked_clocks += _blocked_counter
            if _blocked_counter > 0:
                blocked = True
        elif type(node) is QueueSystem.Server:
            total_discarded += node.get_discarded_counter()
            total_token_load += node.get_busy_rate()
            average_serve_duration += \
                (node.get_served_counter() + node.get_blocked_counter()) / node.get_served_counter()

            _blocked_counter = node.get_blocked_counter()
            total_blocked_clocks += _blocked_counter
            if _blocked_counter > 0:
                blocked = True
        elif type(node) is QueueSystem.Queue:
            average_queue_len += node.get_average_load()
            average_queue_wait += node.get_average_wait()
            average_serve_duration += node.get_average_wait()
            total_token_load += node.get_average_load() * node.get_busy_rate()
        average_busy_rate += node.get_busy_rate()
        if blocked:
            total_blocked_nodes += 1
    for node in terminals:
        if type(node) is QueueSystem.Server:
            total_tokens_served += node.get_served_counter()

    if total_blocked_nodes > 0:
        average_block_rate = total_blocked_clocks / nodes[len(nodes) - 1].get_tick_counter() / total_blocked_nodes
    else:
        average_block_rate = 0.0
    if len(nodes) > 0:
        average_busy_rate /= len(nodes)
    if total_queue_nodes > 0:
        average_queue_len /= total_queue_nodes
        average_queue_wait /= total_queue_nodes
    if nodes[len(nodes) - 1].get_tick_counter() > 0:
        average_tokens_served = total_tokens_served / nodes[len(nodes) - 1].get_tick_counter()
    else:
        average_tokens_served = 0
    if total_tokens_generated > 0:
        total_serve_rate = total_tokens_served / total_tokens_generated
        total_discard_rate = total_discarded / total_tokens_generated
    else:
        total_serve_rate = 0
        total_discard_rate = 0

    print(f"Total token serve probability: {total_serve_rate}")
    print(f"Average tokens served per clock: {average_tokens_served}")
    print(f"System deny probability: {total_discard_rate}")
    print(f"System block probability: {average_block_rate}")
    print(f"Average queue length: {average_queue_len}")
    print(f"Average simultaneously processed tokens: {total_token_load}")
    print(f"Average queue token wait duration: {average_queue_wait}")
    print(f"Average token service duration: {average_serve_duration}")
    print(f"System load rate: {average_busy_rate}")


if __name__ == '__main__':
    while True:
        lab()
        print(f"Quit? [y]/n")
        if input() != "n":
            break
