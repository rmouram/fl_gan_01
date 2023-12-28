## Geração de dados sintéticos via GAN utilizando aprendizado federado personalizado

- Geração de dados do MNIST: necessário avaliar se manterá assim ou será trocada a NN.
- Discriminador está no cliente e a agregação que ocorre no servidor é apenas do gerador, é necessário avaliar se isso acarreta problemas.
- Caso seja necessário agregar também os pesos do discriminador, alterar a função de agregação para suportar a mudança.

