from pathlib import Path
from typing import Tuple, List
import implicit
import scipy

from data import load_user_artists, ArtistRetriever


class ImplicitRecommender:
    """
    This class computes recommendation for a given user
    using the implicit library.

    Attributes:
        -artist_retriever: ArtistRetriever instance
        -implicit_model : an implicit model
    """

    def __init__(self, artist_receiver: ArtistRetriever, implicit_model: implicit.recommender_base.RecommenderBase):
        self.artist_receiver = artist_receiver
        self.implicit_model = implicit_model

    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix) -> None:
        """
        Fit the model to the user artists matrix
        :param user_artists_matrix:
        :return: None
        """
        self.implicit_model.fit(user_artists_matrix)

    def recommend(self, user_id: int, user_artists_matrix: scipy.sparse.csr_matrix, n: int = 10) -> Tuple[
        List[str], List[float]]:
        artists_ids, scores = self.implicit_model.recommend(user_id, user_artists_matrix[n])
        artists = [self.artist_receiver.get_artist_name_from_id(artist_id) for artist_id in artists_ids]
        return artists, scores

if __name__ == "__main__":
    user_artists = load_user_artists(Path("dataset/user_artists.dat"))

    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("dataset/artists.dat"))

    implicit_model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01
    )

    recommender = ImplicitRecommender(artist_retriever, implicit_model)
    recommender.fit(user_artists)
    artists,scores=recommender.recommend(3,user_artists,n=5)
    for artist,score in zip(artists,scores):
        print(f"{artist}: {score}")
