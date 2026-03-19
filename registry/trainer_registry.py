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
        return cls._trainers.get(name)