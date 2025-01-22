
import importlib


ensemble_logits_processor = {
    "qualname": "vital_llm_reasoner_server.ensemble_server.LoggingLogitsProcessor"
}


def test_dynamic_instantiation(qualname):
    module_name, class_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    instance = cls()
    print(f"Successfully instantiated: {instance}")
    return instance


test_dynamic_instantiation("vital_llm_reasoner_server.ensemble_server.LoggingLogitsProcessor")
