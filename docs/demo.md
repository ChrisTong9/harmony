# Audio demo

The [Harmony project website](https://christong9.github.io/harmony/) presents the abstract and audio comparisons for addition, removal, replacement, and volume adjustment.

The website source and audio assets are maintained on the [`gh-pages` branch](https://github.com/ChrisTong9/harmony/tree/gh-pages). GitHub Pages publishes that branch from its root. The `main` branch contains the editing implementation and the 420-example CSV list.

To preview the website in a separate checkout:

```bash
git clone --branch gh-pages --single-branch https://github.com/ChrisTong9/harmony.git harmony-demo
cd harmony-demo
python -m http.server 8766 --bind 127.0.0.1
```

Open `http://localhost:8766/`, or forward port 8766 when using SSH.
