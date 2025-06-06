### **Step-by-Step Guide to Create Product Categories, Attributes, and Basic Products in Aimeos**

Setting up an online store in Aimeos involves organizing your products into categories, defining attributes, and adding the actual products. Here’s a detailed guide:

---

### **1. Access the Aimeos Admin Panel**
1. Start your Laravel server:
   ```bash
   php artisan serve
   ```
2. Navigate to the admin panel:
   ```
   http://localhost:8000/admin
   ```
3. Log in using the admin credentials you configured during setup.

---

### **2. Create Product Categories**
Categories help organize your products into groups for easier navigation.

#### **Steps:**
1. **Navigate to Categories:**
   - In the admin panel, go to **Catalog → Categories**.

2. **Add a New Category:**
   - Click the **+ Add** button.
   - Fill in the details:
     - **Name:** The name of the category (e.g., "Electronics").
     - **Code:** A unique identifier for internal use (e.g., "electronics").
     - **Type:** Choose "Default" for standard categories.
     - **Status:** Set to "Enabled" to make it active.

3. **Add Images (Optional):**
   - Use the **Media** tab to upload an image for the category (e.g., category banner).

4. **Save the Category:**
   - Click **Save** to store the new category.

---

### **3. Define Product Attributes**
Attributes allow you to describe product features (e.g., color, size).

#### **Steps:**
1. **Navigate to Attributes:**
   - In the admin panel, go to **Product → Attributes**.

2. **Add a New Attribute:**
   - Click the **+ Add** button.
   - Fill in the details:
     - **Name:** The name of the attribute (e.g., "Color").
     - **Code:** A unique identifier (e.g., "color").
     - **Type:** Select the appropriate type (e.g., "List" for dropdown values).

3. **Add Attribute Options:**
   - After creating the attribute, click on it to open the details.
   - Go to the **Options** tab and add options (e.g., "Red," "Blue," "Green").

4. **Save the Attribute:**
   - Click **Save** to finalize.

---

### **4. Add a Basic Product**
Once categories and attributes are set, you can add products.

#### **Steps:**
1. **Navigate to Products:**
   - In the admin panel, go to **Product → Products**.

2. **Add a New Product:**
   - Click the **+ Add** button.
   - Fill in the details:
     - **Name:** The product name (e.g., "Smartphone").
     - **Code:** A unique identifier (e.g., "smartphone-001").
     - **Type:** Choose "Default" for basic products.
     - **Status:** Set to "Enabled" to make it available in the store.

3. **Assign a Category:**
   - Go to the **Catalogs** tab.
   - Select the relevant category (e.g., "Electronics") to associate the product.

4. **Set Attributes:**
   - Go to the **Attributes** tab.
   - Assign relevant attributes (e.g., "Color: Red").

5. **Set Pricing:**
   - Go to the **Price** tab and add:
     - **Price:** Set the product price (e.g., "299.99").
     - **Tax Rate:** Define the applicable tax (e.g., "18%").

6. **Add Images (Optional):**
   - Go to the **Media** tab to upload product images.

7. **Set Stock Levels:**
   - Go to the **Stock** tab and configure:
     - **Quantity:** Specify the available stock.
     - **Low Stock Level:** Define when to trigger low-stock notifications.

8. **Save the Product:**
   - Click **Save** to store the new product.

---

### **5. Verify the Setup**
1. Visit the **Frontend Store**:  
   Navigate to `http://localhost:8000/shop` to view your products and categories.  
2. Check the product listing and ensure attributes like color are visible.  
3. Test the category navigation to verify proper organization.

---

### **Tips:**
- Use descriptive names and codes for easier management.
- Regularly update stock levels to reflect inventory changes.
- Optimize product images for faster loading times on the frontend.

Would you like to explore advanced product configurations or other features?