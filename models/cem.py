"""
Concept Embedding Model (CEM) implementation.

Based on: "Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off"
by Espinosa Zarlenga et al. (NeurIPS 2022)

Key idea: Each concept is represented by a d-dimensional embedding (mixture of
learned positive/negative embeddings weighted by predicted concept probability),
rather than a single scalar as in standard CBMs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cub.config import N_ATTRIBUTES_CBM, N_CLASSES


class ConceptEmbeddingModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        n_concepts: int = N_ATTRIBUTES_CBM,
        n_classes: int = N_CLASSES,
        emb_size: int = 16,
        training_intervention_prob: float = 0.25,
        embedding_activation: str = "leakyrelu",
        shared_prob_gen: bool = True,
        c2y_layers: list = None,
    ):
        """
        Args:
            backbone: Feature extractor (Inception3, DINO, etc.)
            n_concepts: Number of binary concepts (112 for CUB-CBM)
            n_classes: Number of output classes (200 for CUB)
            emb_size: Dimensionality of each concept embedding
            training_intervention_prob: Probability of random concept intervention during training
            embedding_activation: Activation for context generators ("leakyrelu" or "relu")
            shared_prob_gen: If True, all concepts share one probability generator
            c2y_layers: Hidden layer sizes for c2y MLP. If None, uses [n_concepts * emb_size].
        """
        super().__init__()
        self.backbone = backbone
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.training_intervention_prob = training_intervention_prob

        # Pool backbone features to a vector
        self.pool = nn.AdaptiveAvgPool2d(1)

        backbone_dim = backbone.final_channel_dim

        # Activation for context generators
        if embedding_activation == "leakyrelu":
            act_fn = nn.LeakyReLU
        elif embedding_activation == "relu":
            act_fn = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {embedding_activation}")

        # One context generator per concept: maps backbone features -> (pos_emb, neg_emb)
        self.concept_context_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_dim, 2 * emb_size),
                act_fn(),
            )
            for _ in range(n_concepts)
        ])

        # Probability generators: predict concept probability from context
        if shared_prob_gen:
            self.shared_prob_gen = True
            self.concept_prob_generator = nn.Sequential(
                nn.Linear(2 * emb_size, 1),
            )
        else:
            self.shared_prob_gen = False
            self.concept_prob_generators = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * emb_size, 1),
                )
                for _ in range(n_concepts)
            ])

        # Concept-to-class MLP
        bottleneck_dim = n_concepts * emb_size
        if c2y_layers is None:
            c2y_layers = []

        layers = []
        prev_dim = bottleneck_dim
        for hidden_dim in c2y_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.c2y_model = nn.Sequential(*layers)

    def _generate_concept_embeddings(self, features):
        """Generate per-concept context embeddings and predicted probabilities.

        Args:
            features: Pooled backbone features [B, D]
        Returns:
            contexts: [B, n_concepts, 2 * emb_size]
            c_probs: [B, n_concepts] concept probabilities (after sigmoid)
        """
        contexts = []
        c_probs = []

        for i, context_gen in enumerate(self.concept_context_generators):
            context_i = context_gen(features)  # [B, 2 * emb_size]
            contexts.append(context_i)

            if self.shared_prob_gen:
                prob_i = torch.sigmoid(self.concept_prob_generator(context_i))  # [B, 1]
            else:
                prob_i = torch.sigmoid(self.concept_prob_generators[i](context_i))  # [B, 1]
            c_probs.append(prob_i)

        contexts = torch.stack(contexts, dim=1)  # [B, n_concepts, 2*emb_size]
        c_probs = torch.cat(c_probs, dim=1)  # [B, n_concepts]

        return contexts, c_probs

    def _mix_embeddings(self, contexts, c_probs):
        """Mix positive and negative embeddings based on concept probabilities.

        Args:
            contexts: [B, n_concepts, 2 * emb_size]
            c_probs: [B, n_concepts]
        Returns:
            bottleneck: [B, n_concepts * emb_size]
        """
        pos_emb = contexts[:, :, :self.emb_size]   # [B, n_concepts, emb_size]
        neg_emb = contexts[:, :, self.emb_size:]    # [B, n_concepts, emb_size]

        prob = c_probs.unsqueeze(-1)  # [B, n_concepts, 1]
        mixed = pos_emb * prob + neg_emb * (1 - prob)  # [B, n_concepts, emb_size]

        return mixed.view(mixed.size(0), -1)  # [B, n_concepts * emb_size]

    def _apply_interventions(self, c_probs, c_true):
        """During training, randomly intervene on concepts (replace predicted with GT).

        Args:
            c_probs: [B, n_concepts] predicted concept probabilities
            c_true: [B, n_concepts] ground truth concept labels
        Returns:
            c_probs: [B, n_concepts] with some concepts replaced by GT
        """
        if not self.training or self.training_intervention_prob <= 0:
            return c_probs

        # Sample which concepts to intervene on
        mask = torch.bernoulli(
            torch.full((self.n_concepts,), self.training_intervention_prob, device=c_probs.device)
        )  # [n_concepts]
        mask = mask.unsqueeze(0).expand_as(c_probs)  # [B, n_concepts]

        return c_probs * (1 - mask) + c_true * mask

    def forward(self, x, attr_labels=None):
        """
        Args:
            x: Input images [B, 3, H, W]
            attr_labels: Ground truth concept labels [B, n_concepts] (used for training interventions)
        Returns:
            Tuple of (class_logits, concept_scores, None)
            - class_logits: [B, n_classes]
            - concept_scores: [B, n_concepts] (concept probabilities, already sigmoided)
            - None: placeholder for attention maps (CEM has no spatial maps)
        """
        # Extract backbone features
        if self.training and hasattr(self.backbone, 'aux_logits') and self.backbone.aux_logits:
            features, _ = self.backbone(x)
        else:
            features = self.backbone(x)

        # Pool to vector
        features = self.pool(features)
        features = features.view(features.size(0), -1)  # [B, D]

        # Generate concept embeddings and probabilities
        contexts, c_probs = self._generate_concept_embeddings(features)

        # Apply random interventions during training
        if attr_labels is not None:
            c_probs_mixed = self._apply_interventions(c_probs, attr_labels)
        else:
            c_probs_mixed = c_probs

        # Mix positive/negative embeddings based on (possibly intervened) probabilities
        bottleneck = self._mix_embeddings(contexts, c_probs_mixed)

        # Predict class from bottleneck
        class_logits = self.c2y_model(bottleneck)

        # Return in the same format as other models: (output, scores, maps)
        # c_probs are already in [0,1] (post-sigmoid), so we convert to logits for compatibility
        # with eval.py which expects raw scores and applies sigmoid itself
        concept_logits = torch.log(c_probs / (1 - c_probs + 1e-7) + 1e-7)

        return (class_logits, concept_logits, None)


def build_cem(args) -> ConceptEmbeddingModel:
    """Factory function to build a CEM from config args."""
    from models.models import backbone_by_name

    backbone = backbone_by_name(args)

    emb_size = getattr(args, "cem_emb_size", 16)
    training_intervention_prob = getattr(args, "cem_training_intervention_prob", 0.25)
    embedding_activation = getattr(args, "cem_embedding_activation", "leakyrelu")
    shared_prob_gen = getattr(args, "cem_shared_prob_gen", True)

    c2y_layers_raw = getattr(args, "cem_c2y_layers", None)
    c2y_layers = list(c2y_layers_raw) if c2y_layers_raw else None

    return ConceptEmbeddingModel(
        backbone=backbone,
        n_concepts=args.n_attributes,
        n_classes=N_CLASSES,
        emb_size=emb_size,
        training_intervention_prob=training_intervention_prob,
        embedding_activation=embedding_activation,
        shared_prob_gen=shared_prob_gen,
        c2y_layers=c2y_layers,
    )
