
# Define a wrapper function to directly add functions to different registries
def add_to_registry(registry, name):
    def wrapper(func):
        registry[name] = func
        return func
    return wrapper