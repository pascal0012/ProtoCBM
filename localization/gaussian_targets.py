import torch


class GaussianTargetGenerator:
    """
    Generates 2D Gaussian heatmap targets centered on ground truth keypoints,
    at the spatial resolution of attention maps.
    """

    def __init__(self, img_size: int = 299, sigma: float = 1.0):
        """
        Args:
            img_size: Size of the input images (assumes square images)
            sigma: Standard deviation of the Gaussian in feature map space
        """
        self.img_size = img_size
        self.sigma = sigma
        self._coord_grid = None

    def _build_coord_grid(self, H, W, device):
        """Pre-compute coordinate grid [H, W, 2] for Gaussian generation."""
        ys = torch.arange(H, dtype=torch.float32, device=device)
        xs = torch.arange(W, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self._coord_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

    def generate(
        self,
        part_gts: torch.Tensor,
        part_indices: torch.Tensor,
        H: int,
        W: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate Gaussian heatmap targets and validity masks.

        Args:
            part_gts: [B, K, 2] ground truth keypoint coordinates (x, y) in image space
            part_indices: [M] part index for each mapped attribute
            H: Height of the attention/feature map
            W: Width of the attention/feature map

        Returns:
            target: [B, M, H, W] Gaussian heatmap targets
            valid_mask: [B, M] binary mask (1 for visible parts, 0 for invisible)
        """
        device = part_gts.device

        # Lazily build coordinate grid
        if self._coord_grid is None or self._coord_grid.shape[0] != H or self._coord_grid.shape[1] != W:
            self._build_coord_grid(H, W, device)
        coord_grid = self._coord_grid  # [H, W, 2]

        # Scale GT keypoints from image space to feature map space
        scale = torch.tensor([W / self.img_size, H / self.img_size], device=device)
        scaled_gts = part_gts.float() * scale  # [B, K, 2]

        # Gather GT centers for all mapped attributes: [B, M, 2]
        centers = scaled_gts[:, part_indices, :]  # [B, M, 2]

        # Build Gaussian targets: [B, M, H, W]
        diff = coord_grid.unsqueeze(0).unsqueeze(0) - centers[:, :, None, None, :]
        target = torch.exp(-diff.pow(2).sum(-1) / (2 * self.sigma ** 2))  # [B, M, H, W]

        # Validity mask: parts with GT [0, 0] (invisible) should be excluded
        gt_for_mapped = part_gts[:, part_indices, :].float()  # [B, M, 2]
        valid_mask = (gt_for_mapped.abs().sum(dim=-1) > 0).float()  # [B, M]

        return target, valid_mask
