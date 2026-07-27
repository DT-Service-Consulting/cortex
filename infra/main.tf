terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }

  backend "azurerm" {
    resource_group_name  = "rg-tfstate"
    storage_account_name = "stcortexb4074d"
    container_name       = "tfstate"
    key                  = "cortex.tfstate"
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "main" {
  name     = "rg-cortex"
  location = "westeurope"
}

resource "azurerm_storage_account" "datalake" {
  name                     = "stcortexdlake"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  is_hns_enabled           = true
}

resource "azurerm_storage_container" "private" {
  name                  = "private"
  storage_account_id    = azurerm_storage_account.datalake.id
  container_access_type = "private"
}

resource "azurerm_storage_container" "opendata" {
  name                  = "public"
  storage_account_id    = azurerm_storage_account.datalake.id
  container_access_type = "private"
}
