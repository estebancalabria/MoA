import json
import sys

#print("Cargando Configuracion del archivo config.json")

class Config:
    endpoint = 'http://localhost:8080'
    models = []
    main_model = ""
    api_key = None


try:
    with open("config.json", 'r') as f:
        diccionario =  json.load(f)
        
        if "endpoint" in diccionario:
            Config.endpoint = diccionario['endpoint']
        else:
            sys.exit("Error: 'endpoint' key not found in config.json")
        
        if "models" in diccionario:
            Config.models = diccionario['models']
        else:
            sys.exit("Error: 'models' key not found in config.json")

        if "mainModel" in diccionario:
            Config.main_model = diccionario['mainModel']
        else:
            sys.exit("Error: 'mainModel' key not found in config.json")

        if "api_key" in diccionario:
            Config.api_key = diccionario['api_key']


except FileNotFoundError:
    # Lanza una excepción personalizada si el archivo no se encuentra
    print("Error: config.json file not found")
    sys.exit("El archivo config.json debe existir")
except json.JSONDecodeError:
    # Lanza una excepción personalizada si el archivo no es JSON válido
    print("Error: config.json file is not a valid JSON")
    sys.exit("El archivo config.json debe ser un JSON válido")


#print(Config)

"""
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
    
class Config(metaclass=SingletonMeta):
    _settings = {}
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, filename='config.json'):
            self._settings = self._load_settings(filename)
            print(f"Settings: {self._settings}")

    def _load_settings(self, filename):
        print(f"Loading settings from {filename}")
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"File {filename} not found. Using default settings.")
            return {}
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {filename}. Using default settings.")
            return {}

    def __getattr__(self, name):
        return self._settings.get(name)
        #raise AttributeError(f"'Config' object has no attribute '{name}'")

c = Config()
print(c._settings.get('endpoint'))
print(c.endpoint)
print(Config.endpoint)
"""