import implicit

from .mf_implicit_model import MFImplicitModel


class ALS(MFImplicitModel):
    def __init__(self, latent_dim, reg_term, epochs, random_state=42, **kwargs):
        super(ALS, self).__init__(
            latent_dim=latent_dim, reg_term=reg_term, epochs=epochs, learning_rate=None
        )

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.latent_dim,
            regularization=self.reg_term,
            iterations=self.epochs,
            random_state=random_state
        )
