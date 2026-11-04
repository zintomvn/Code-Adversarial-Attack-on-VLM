from .common import *
from .config_schema import cfg
from .utils import V53Store

class V53Preview:
    @staticmethod
    def show(shared, variant_folder, step=1):
        import matplotlib.pyplot as plt
        shared, variant_folder = Path(shared), Path(variant_folder)
        views = variant_folder / "views" / f"step_{step:03d}"
        entries = [("source", shared/"source.png"), ("original target", shared/"target.png"),
                   ("effective target", shared/"effective_target.png"),
                   ("adv input BEFORE update", views/"input_adv.png"),
                   *[(f"actual crop {i}", views/f"crop_{i}.png") for i in range(5)],
                   ("adv AFTER update", variant_folder/f"step_{step:03d}.png")]
        fig, axes = plt.subplots(3, 4, figsize=(15, 11))
        for axis in axes.flat:
            axis.axis("off")
        for axis, (title, path) in zip(axes.flat, entries):
            axis.set_title(title)
            if path.exists():
                with Image.open(path) as handle:
                    axis.imshow(handle.convert("RGB"))
            else:
                axis.text(.1, .5, "PENDING artifact")
        source_path, adv_path = shared/"source.png", variant_folder/f"step_{step:03d}.png"
        if source_path.exists() and adv_path.exists():
            with Image.open(source_path) as s, Image.open(adv_path) as a:
                delta = np.asarray(a).astype(float)-np.asarray(s).astype(float)
            axes.flat[10].imshow(np.clip(.5+delta/(2*cfg.V53_EPSILON), 0, 1))
            axes.flat[10].set_title("signed delta (display scaled)")
        provenance = shared/"diffusion.json"
        if provenance.exists():
            record = V53Store.read(provenance)
            fig.suptitle(record["effective_target"]+" | prompt: "+record.get("prompt", ""), fontsize=10)
        coordinates = views/"record.json"
        if coordinates.exists() and (views/"input_adv.png").exists():
            from matplotlib.patches import Rectangle
            with Image.open(views/"input_adv.png") as handle:
                axes.flat[11].imshow(handle)
            for index, (top, left, height, width) in enumerate(V53Store.read(coordinates)["boxes"]):
                axes.flat[11].add_patch(Rectangle((left, top), width, height, fill=False,
                                                edgecolor=plt.cm.tab10(index), linewidth=1))
            axes.flat[11].set_title("5 actual crop boxes")
        fig.tight_layout()
        output = variant_folder / f"preview_step_{step:03d}.png"
        fig.savefig(output, dpi=120)
        plt.close(fig)
        return output
