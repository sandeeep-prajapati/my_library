Writing a basic Terraform configuration to provision infrastructure on cloud providers like AWS, GCP, or Azure involves defining resources in HCL (HashiCorp Configuration Language) format. Below are examples for each cloud provider to create a simple virtual machine (EC2 instance for AWS, Compute Engine for GCP, and Virtual Machine for Azure).

### Example 1: AWS EC2 Instance

1. **Create a directory for your Terraform configuration**:
   ```bash
   mkdir terraform-aws
   cd terraform-aws
   ```

2. **Create a file named `main.tf`**:
   ```hcl
   provider "aws" {
     region = "us-east-1"
   }

   resource "aws_instance" "my_instance" {
     ami           = "ami-0c55b159cbfafe1f0"  # Amazon Linux 2 AMI (HVM), SSD Volume Type
     instance_type = "t2.micro"

     tags = {
       Name = "MyInstance"
     }
   }
   ```

3. **Initialize Terraform**:
   ```bash
   terraform init
   ```

4. **Create an execution plan**:
   ```bash
   terraform plan
   ```

5. **Apply the configuration**:
   ```bash
   terraform apply
   ```

### Example 2: GCP Compute Engine Instance

1. **Create a directory for your Terraform configuration**:
   ```bash
   mkdir terraform-gcp
   cd terraform-gcp
   ```

2. **Create a file named `main.tf`**:
   ```hcl
   provider "google" {
     project = "<YOUR_PROJECT_ID>"
     region  = "us-central1"
   }

   resource "google_compute_instance" "my_instance" {
     name         = "my-instance"
     machine_type = "f1-micro"
     zone         = "us-central1-a"

     boot_disk {
       initialize_params {
         image = "debian-cloud/debian-9"
       }
     }

     network_interface {
       network = "default"

       access_config {
         // Include this to give the instance a public IP address
       }
     }
   }
   ```

3. **Initialize Terraform**:
   ```bash
   terraform init
   ```

4. **Create an execution plan**:
   ```bash
   terraform plan
   ```

5. **Apply the configuration**:
   ```bash
   terraform apply
   ```

### Example 3: Azure Virtual Machine

1. **Create a directory for your Terraform configuration**:
   ```bash
   mkdir terraform-azure
   cd terraform-azure
   ```

2. **Create a file named `main.tf`**:
   ```hcl
   provider "azurerm" {
     features {}
   }

   resource "azurerm_resource_group" "example" {
     name     = "example-resources"
     location = "East US"
   }

   resource "azurerm_virtual_network" "example" {
     name                = "example-network"
     address_space       = ["10.0.0.0/16"]
     location            = azurerm_resource_group.example.location
     resource_group_name = azurerm_resource_group.example.name
   }

   resource "azurerm_subnet" "example" {
     name                 = "example-subnet"
     resource_group_name  = azurerm_resource_group.example.name
     virtual_network_name = azurerm_virtual_network.example.name
     address_prefixes     = ["10.0.1.0/24"]
   }

   resource "azurerm_network_interface" "example" {
     name                = "example-nic"
     location            = azurerm_resource_group.example.location
     resource_group_name = azurerm_resource_group.example.name

     ip_configuration {
       name                          = "internal"
       subnet_id                    = azurerm_subnet.example.id
       private_ip_address_allocation = "Dynamic"
     }
   }

   resource "azurerm_linux_virtual_machine" "example" {
     name                = "example-vm"
     resource_group_name = azurerm_resource_group.example.name
     location            = azurerm_resource_group.example.location
     size                = "Standard_DS1_v2"
     admin_username      = "adminuser"
     admin_password      = "P@ssw0rd1234!"

     network_interface_ids = [
       azurerm_network_interface.example.id,
     ]

     os_disk {
       caching              = "ReadWrite"
       create_option        = "FromImage"
     }

     source_image_reference {
       publisher = "Canonical"
       offer     = "UbuntuServer"
       sku       = "18.04-LTS"
       version   = "latest"
     }
   }
   ```

3. **Initialize Terraform**:
   ```bash
   terraform init
   ```

4. **Create an execution plan**:
   ```bash
   terraform plan
   ```

5. **Apply the configuration**:
   ```bash
   terraform apply
   ```

### Summary

In each of these examples, we defined a simple configuration to create a virtual machine in the respective cloud provider. Here's a quick recap of the steps:

1. Create a directory for your configuration files.
2. Create a `main.tf` file with the necessary provider and resource definitions.
3. Initialize Terraform with `terraform init`.
4. Review the plan with `terraform plan`.
5. Apply the configuration with `terraform apply`.
