terraform {
  backend "gcs" {
    bucket = "miad-paad-rs-tfstate-dev"
    prefix = "terraform/dev"
  }
}