# Adding a New Garment Sticker

Each sticker entry needs **3 files**:

| File | Purpose |
|------|---------|
| `beforeImg` | Original person photo (left side of slider) |
| `afterImg` | AI try-on result photo (right side of slider) |
| `img` (sticker icon) | The garment photo, background-removed |

---

## Step 1 — Remove the background from the garment photo

```bash
cd C:/Projects/HooHacks26
venv/Scripts/python.exe -c "
from rembg import remove
from PIL import Image
import io

with open('finalimages/YOUR_GARMENT.jpg', 'rb') as f:
    data = f.read()

result = remove(data)
img = Image.open(io.BytesIO(result))
img.save('envision/public/YOUR_GARMENT_sticker.png')
print(img.size)
"
```

## Step 2 — Copy files to the right places

```bash
# Sticker PNG is already in envision/public/ from Step 1

# Before/after photos → frontend public folder
cp finalimages/BEFORE_PHOTO.jpg  envision/public/before_name.jpg
cp finalimages/AFTER_PHOTO.jpg   envision/public/after_name.jpg

# Original garment → backend for pipeline
cp finalimages/YOUR_GARMENT.jpg  backend/garments/garment_name.jpg
```

## Step 3 — Add to `CLOSET_ITEMS` in `envision/src/components/TryOn.tsx`

```ts
{
  id: 12,                          // unique number
  name: 'Display Name',
  img: '/YOUR_GARMENT_sticker.png',
  garment_url: 'http://localhost:8000/garments/garment_name.jpg',
  category: 'tops',                // or 'pants'
  hero: true,
  beforeImg: '/before_name.jpg',
  afterImg: '/after_name.jpg',
},
```

That's it. The sticker appears in the closet, clicking it switches the before/after comparison slider on the left panel.
