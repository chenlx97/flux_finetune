class TrainerRegistry:
    _trainers = {}

    @classmethod
    def register(cls, name):
        def wrapper(trainer_cls):
            cls._trainers[name] = trainer_cls
            return trainer_cls
        return wrapper

    @classmethod
    def get(cls, name):
        if name not in cls._trainers:
            raise ValueError(f"Trainer {name} not registered. Available: {list(cls._trainers.keys())}")
        return cls._trainers[name]