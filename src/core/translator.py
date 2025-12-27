# Não estamos mais utilizando o numpy como gambiarra!

class Translator:
    """
    Responsável por converter o rótulo do sinal (string) em dados de saída (texto/áudio).
    Não lida com detecção visual ou Machine Learning.
    """
    def __init__(self):
        self.maps_libras_to_text = self._load_mappings()

    def translate(self, signal: str) -> str:
        """
        Recebe o sinal (string) confirmado pelo SignalBuffer e retorna sua tradução textual.
        """
        if not signal or signal in ["Desconhecido", "Nenhum"]:
            return ""

        # Retorna o valor mapeado ou o próprio sinal, caso não exista um mapeamento mais complexo.
        return self.maps_libras_to_text.get(signal, signal)

    def _load_mappings(self):
        """
        Mapeamento de rótulos para significado final (expansível).
        """
        return {
            "A": "Letra A (Tradução)",
            "B": "Letra B (Tradução)",
            "L": "Letra L (Tradução)",
            "D": "Letra D (Tradução)",
            # Mapeamentos para palavras completas virão aqui no futuro.
        }