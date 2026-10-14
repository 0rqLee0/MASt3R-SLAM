

# ---------------------------------------------------------------------------
# plane_vio baseline: generic image folder exported by extract_images.py
#   <dir>/calib.json + <dir>/cam0/<stamp_ns>.png (+ <dir>/cam0_pinhole for TUM-VI fisheye)
# Images are always undistorted (radtan via cv2, like the upstream EuRoC loader); intrinsics are
# used for optimisation only when use_calib is on. Timestamps are written in seconds.
# ---------------------------------------------------------------------------
class CalibJsonDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        import json
        self.use_calibration = True
        self.dataset_path = pathlib.Path(dataset_path)
        c = json.load(open(self.dataset_path / "calib.json"))
        if (self.dataset_path / "cam0_pinhole").is_dir() and "cam0_pinhole" in c:
            img_dir = self.dataset_path / "cam0_pinhole"
            intr = list(c["cam0_pinhole"]["intrinsics"])
            W, H = c["cam0_pinhole"]["resolution"]
            dist = []
        else:
            if c.get("distortion_model", "radtan") != "radtan":
                raise ValueError("unsupported distortion model " + c.get("distortion_model"))
            img_dir = self.dataset_path / "cam0"
            intr = list(c["intrinsics"])
            W, H = c["resolution"]
            dist = list(c["distortion_coeffs"])
        files = sorted(img_dir.glob("*.png"), key=lambda p: int(p.stem))
        self.rgb_files = files
        self.timestamps = np.array(
            [f"{int(p.stem) // 10**9}.{int(p.stem) % 10**9:09d}" for p in files]
        )
        self.camera_intrinsics = Intrinsics.from_calib(
            self.img_size, W, H, [*intr, *dist], always_undistort=True
        )
        print(f"[CalibJsonDataset] {img_dir} n={len(files)} W={W} H={H} calib={intr} dist={dist}")

    def read_img(self, idx):
        img = cv2.imread(str(self.rgb_files[idx]), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


_load_dataset_upstream = load_dataset


def load_dataset(dataset_path):
    if (pathlib.Path(dataset_path) / "calib.json").exists():
        return CalibJsonDataset(dataset_path)
    return _load_dataset_upstream(dataset_path)
