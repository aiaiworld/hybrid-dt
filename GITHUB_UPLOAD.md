# Dua artifact len GitHub

Thu muc nay da la mot repository doc lap ve noi dung, nhung chua duoc `git
init` de khong lam thay doi repository cha tren may.

## Trong giai doan double-blind

Khong dua link tu tai khoan GitHub ca nhan vao ban thao. Co the:

1. nop truc tiep thu muc nay duoi dang supplementary material;
2. dung mot dich vu anonymous GitHub;
3. tao private repository va chi cong khai sau khi ket thuc review.

Metadata hien tai dung ten `Anonymous Authors`.

## Khoi tao repository

```bash
cd "hybrid-dt-reproducibility"
git init -b main
git add .
git status --short
git commit -m "Release Hybrid-DT reproducibility artifact"
git remote add origin <REPOSITORY_URL>
git push -u origin main
```

`outputs/`, virtual environments, Python cache, and local editor files are
excluded by `.gitignore`. The 2.5 MB compressed TelecomTS snapshot is included
because it pins the exact records used in the paper.

## Kiem tra truoc khi push

```bash
python -m unittest discover -s tests -v
./scripts/reproduce_paper.sh
git status --short
```

Sau khi bai duoc phep de-anonymize, cap nhat tac gia trong:

- `CITATION.cff`;
- `LICENSE`;
- `pyproject.toml`;
- phan citation trong `README.md`.
