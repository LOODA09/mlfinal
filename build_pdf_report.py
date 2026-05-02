from hotel_app import BenchmarkPdfBuilder


if __name__ == "__main__":
    path = BenchmarkPdfBuilder("artifacts").build()
    print(path)
