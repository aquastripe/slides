# Marp Pages

Marp Pages is a static site generator with Marp. You can:
- Make slides using Markdown syntax.
- Publish slide decks on GitHub Pages automatically with GitHub Actions.

## Usage

### Prerequisite

- [Node.js](https://nodejs.org/)
- Install dependency packages:
    ```bash
    npm i
    ```
- Python 3.6+

Make slide decks and put them in the `content/` directory. 
This work is based on [Marp](https://marp.app/), your slides should follow its settings and dialects.

### Preview

```bash
npm run preview
```

### Publish on GitHub Pages

1. Commit and push to your GitHub repository.
2. Setup GitHub Pages and change the branch to `gh-pages`:
    1. Navigate to **Settings/Pages**
    2. Select **Branch** to `gh-pages` and save.

### Update from this template

1. setup remote repository (once)
    ```bash
    git remote add template git@github.com:aquastripe/marp-pages.git
    ```
2. update if there are changes in this template
    ```bash
    git fetch template main
    ```
3. merge them
    ```bash
    git merge template/main --allow-unrelated-histories
    ```
