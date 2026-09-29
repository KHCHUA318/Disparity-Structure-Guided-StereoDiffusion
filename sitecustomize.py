try:
    import huggingface_hub

    if not hasattr(huggingface_hub, "cached_download"):
        from huggingface_hub import hf_hub_download

        def cached_download(*args, **kwargs):
            return hf_hub_download(*args, **kwargs)

        huggingface_hub.cached_download = cached_download

    print("sitecustomize: huggingface_hub cached_download shim ready")

except Exception as e:
    print("sitecustomize warning:", e)

