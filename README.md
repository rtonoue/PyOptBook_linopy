# Alternative implementations using linopy

## Overview

This repository provides **alternative implementations** of selected examples
from the book:

> **Pythonではじめる数理最適化  
> 〜ケーススタディでモデリングのスキルを身につけよう〜**（オーム社）

The implementations in this repository are written using
**linopy** instead of **PuLP**, with the goal of demonstrating
a **different modeling approach**.

- Official support repository (PuLP-based):  
  https://github.com/ohmsha/PyOptBook

⚠️ This repository is **not an official support page**.

---

<details>
<summary><strong>Environment setup (uv)</strong></summary>

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

**Prerequisites**

- [uv](https://docs.astral.sh/uv/getting-started/installation/) をインストールする

**手順**

```bash
# リポジトリをクローンしたあと
cd PyOptBook_linopy

# 仮想環境の作成と依存関係のインストール
uv sync
```

- Python は `.python-version` に従い（3.12）、uv が自動で用意します。
- 依存関係は `pyproject.toml` で管理され、`uv.lock` で再現可能です。
- スクリプトや Jupyter を実行するときは、仮想環境を有効化するか `uv run` を使います。

```bash
# 例: 仮想環境を有効化してから実行（Windows PowerShell）
.venv\Scripts\Activate.ps1
jupyter notebook

# または uv run でその場で実行
uv run jupyter notebook
```

</details>

---

## Purpose of this repository

The original repository focuses on:

- clarity and readability for educational purposes
- consistency with the explanations in the book

This repository focuses on:

- expressing similar models using **linopy**
- exploring alternative modeling styles
- providing supplementary examples for readers interested in
  more recent modeling libraries

The intent is **not to improve or replace** the original implementations,
but to offer **additional perspectives** on mathematical modeling in Python.

---

## Notes on performance

Some benchmarks may be included for reference.
However, performance is **not the primary focus** of this repository,
as the examples are intended for educational and illustrative purposes.

---

## Chapters and linopy coverage

| Chapter | Content | linopy 版 |
|--------|--------|-----------|
| 2 | チュートリアル | `2.tutorial/tutorial_linopy.ipynb` + 検証 |
| 3 | 学校のクラス編成 | `3.school/school_linopy.ipynb` + 検証 |
| 4 | クーポン割引（Problem1） | `4.coupon/coupon_linopy.ipynb` + 検証 |
| 5 | 配送計画 | ベースライン・検証スクリプト（`run_baseline_pulp.py`, `verify_linopy.py`）。ルート列挙に数分かかります。 |
| 6 | API 車グループ分け | `6.api/problem_linopy.py`（同一インターフェース）+ 検証。API から `from problem_linopy import CarGroupProblem` で利用可。 |
| 7 | 商品推薦（興味スコア） | **linopy + HiGHS (QP)** で対応。`run_baseline_cvxopt.py` で cvxopt ベースライン取得、`verify_linopy.py` で同一問題を linopy の二次目的で構築し HiGHS で求解・検証。 |

---

## License and attribution

This repository is distributed under the MIT License,
inherited from the original repository.

Please refer to the original repository for the authoritative
implementations and explanations.