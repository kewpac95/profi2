import numpy as np

from profis.utils import load_charset
from profis.rl_fp_reward import decode_indices_to_smiles

def main():
    # Załaduj charset
    charset = load_charset("data/smiles_alphabet.txt")
    print("Charset:", charset)
    print("Długość charsetu:", len(charset))
    print()

    # Sprawdź, czy występują specjalne tokeny
    for special in ['[start]', '[end]', '[nop]']:
        assert special in charset, f"Brak tokena {special} w charset!"

    idx_C = charset.index('C')
    idx_O = charset.index('O')
    idx_end = charset.index('[end]')
    idx_nop = charset.index('[nop]')

    max_len = 10

    # Test 1: ręczny SMILES CCO
    indices = [idx_C, idx_C, idx_O, idx_end] + [idx_nop]*(max_len-4)
    indices = [indices]  # batch size 1
    smiles = decode_indices_to_smiles(indices, charset)
    print("Test 1. Oczekiwane ['CCO']:", smiles)

    # Test 2: [end] w środku sekwencji
    indices2 = [idx_C, idx_end, idx_O, idx_C] + [idx_nop]*(max_len-4)
    indices2 = [indices2]
    smiles2 = decode_indices_to_smiles(indices2, charset)
    print("Test 2. Oczekiwane ['C']:", smiles2)

    # Test 3: losowe indeksy
    np.random.seed(42)
    random_indices = np.random.randint(0, len(charset), size=(3, max_len))
    smiles_list = decode_indices_to_smiles(random_indices, charset)
    print("Test 3. Losowe SMILES:", smiles_list)

    # Test 4: czy model jest w stanie uzyskać [end] token w sampling
    print("\nTest 4. Indeks '[end]' w charset:", idx_end)

if __name__ == "__main__":
    main()