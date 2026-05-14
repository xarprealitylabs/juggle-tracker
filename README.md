# ⚽ Juggle Challenge

POC de brand activation para os mirrors Xarp. Funciona em mobile browser, sem instalação.

## O que faz

- Câmara traseira do telemóvel aponta para a bola
- Detecção por visão (motion tracking) + áudio (peak detection) em paralelo
- Conta keepups em tempo real
- Modo 2 jogadores com handoff do telemóvel
- Modo solo
- Gera clip WebM com overlay do contador para download/share

## Como usar

Abre `index.html` diretamente no browser mobile. Não precisa de servidor.

Para testar localmente:
```bash
npx serve .
# ou
python3 -m http.server 8080
```

## Como funciona a detecção

**Visão (primário):**
- Frame diff entre frames consecutivos → encontra região de movimento
- Rastreia centróide Y do blob em movimento
- Juggle detectado quando bola muda de direção (descida → subida)
- Drop detectado quando não há movimento por 2 segundos

**Áudio (secundário, reforço):**
- Web Audio API AnalyserNode sobre o mic do telemóvel
- Peak >90 (de 255) = som de pontapé → confirma juggle
- Funciona mesmo sem bola visível (e.g. câmara de lado)

## Próximos passos

- [ ] Overlay de score mais elaborado no vídeo final (Canvas text pré-gravação)
- [ ] Leaderboard local (localStorage)
- [ ] Ball detection com COCO-SSD para maior precisão
- [ ] Customização de tema da marca (logo overlay)
- [ ] QR code para abrir no telemóvel a partir do mirror
