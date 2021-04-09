# m7G-DLSTM
Identification of RNA N7-methlguanosine sites in human with directional Double-LSTM Model


Python Requirement:

Python >= 3.0

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Package Requirement:

numpy >= 1.19.5

tensorflow >= 2.4.0

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

sample.txt:
1. RNA sequence in sample.txt will be predicted.
3. Please ensure: (1) each single RNA sequence only occupy one line in the txt file.
                  (2) Sequence string is in upper case.
                  (3) The string only consists of A, C, G and U.
3. Each RNA sequence can have any number of nucleobases (at least 3 nucleobases).
5. All nucleobases will be used for predicting, except for the m7G site.
4. The predicted result could be unauthentic if the number of RNA sequence was not equal to 41.
5. It could take a long time if RNA sequence is too long.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

g_index.txt:
1. Index (begin with 0) of m7G site for each RNA sequence in sample.txt should be stored in this txt file in advance.
2. Each index only occupy one line in the txt file, and it should corresponding to the sample.txt.
3. The index will be unavailable if it is equal the index of the first nucleobase or the last nucleobase.
4. The predicted result could be unauthentic if the m7G site pointed by the index was not in the middle of the RNA sequence.
4. The m7G-DLSTM do not care about what the nucleobase pointed by the index is, but the predicted result could be unauthentic if the m7G site pointed by the index was not G.
5. If this file does not exist, m7G-DLSTM will generate a g_index.txt automatically, and each index will be set as "len(RNA sequence) // 2" (the integer quotient of the length of RNA sequence divided by 2)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

sample.txt example:

AUGCAUUAGCCUUGUGGCUAGAACACCCUCUUCCUACCUCU
UUCUUUUUUUUGUUUCAGAAGAACUGGACGGGGCUGGAGGA
AGGAACCCCCUGAACCCCAAGAGAGGGAGGACCAGGAUCCG
UUUUAGUUAAACGUUGAGGAGAAAAAAAAAAAAGGCUUUUC

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

g_index.txt example:

20

20

20

20

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

probability_of_m7G.txt
1. This file is one of the output files after predicting, that probability of "the nucleobase pointed by index in g_index.txt is m7G site" for each RNA sequence in sample.txt will be written in the file.
2. Each probability only occupy one line in the txt file, and it corresponds to the sample.txt.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

result.txt
1. This file is one of the output files after predicting, that result of "m7G" or "non-m7G" for each RNA sequence in sample.txt will be written in the file.
2. Each result only occupies one line in the txt file, and it corresponds to the sample.txt.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

probability_of_m7G.txt example:

0.34106347

0.8998517

0.7676672

0.76144606

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

result.txt example:

non-m7G

m7G

m7G

m7G

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

After preparing sample.txt and g_index.txt, please running the m7G-DLSTM.exe and waiting for it.
