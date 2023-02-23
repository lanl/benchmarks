# Sphinx Documentation

This folder contains a skeleton that can be used to create Sphinx
documentation. Some relevant links are provided below.

- [How to build documentation](9_appendix/build_docs.rst)

## CI/CD

These scripts have been successfully used to generate documentation hosted as
part of GitLab Pages. The GitLab CI/CD YAML excerpts that achieve this are
provided below for reference.

```yaml
pages:
  stage: deploy
  variables:
    GIT_STRATEGY: none
    WORKSPACE: ""
    PYTHONPATH: "${PYTHONPATH}"
  script:
    - module load anaconda3/2021.05
    - export PATH="/path/to/texlive:${PATH}"
    - pushd doc/sphinx
    - ./build_doc.py --all
    - popd
    - cp -a doc/sphinx/_build/html public
    - touch .nojekyll
    - touch public/.nojekyll
  artifacts:
    paths:
      - public
  tags:
    - relevantbuildsystem
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
```
