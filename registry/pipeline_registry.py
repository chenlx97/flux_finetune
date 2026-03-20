class PipelineRegistry:
    _pipelines = {}

    @classmethod
    def register(cls, name):
        def wrapper(pipeline_cls):
            cls._pipelines[name] = pipeline_cls
            return pipeline_cls
        return wrapper

    @classmethod
    def get(cls, name):
        if name not in cls._pipelines:
            raise ValueError(f"Pipeline {name} not registered. Available: {list(cls._pipelines.keys())}")
        return cls._pipelines[name]