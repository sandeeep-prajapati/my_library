### **How to Create and Configure Products with Multiple Variants (e.g., Size, Color) in Aimeos**

Creating products with variants (like different sizes or colors) is essential for offering diverse choices to customers. Aimeos allows you to manage such products efficiently using attributes and product variations. Here’s a step-by-step guide:

---

### **1. Set Up Variant Attributes**
Attributes define the options for your product variants (e.g., size, color).

#### **Steps:**
1. **Navigate to Attributes:**
   - In the admin panel, go to **Product → Attributes**.

2. **Create Variant Attributes:**
   - Click **+ Add** to create a new attribute.
   - Fill in the details:
     - **Name:** The attribute name (e.g., "Size").
     - **Code:** A unique identifier (e.g., "size").
     - **Type:** Select **List** for attributes with multiple options.

3. **Add Attribute Options:**
   - After creating the attribute, click on it to edit.
   - Go to the **Options** tab.
   - Add specific options for the attribute (e.g., "Small," "Medium," "Large" for size).

4. **Repeat for Other Attributes:**
   - Create additional attributes for other variants (e.g., "Color" with options "Red," "Blue," "Green").

---

### **2. Create the Parent Product**
The parent product acts as a container for all the variant products.

#### **Steps:**
1. **Navigate to Products:**
   - Go to **Product → Products**.

2. **Add the Parent Product:**
   - Click **+ Add** to create a new product.
   - Fill in the details:
     - **Name:** The name of the parent product (e.g., "T-Shirt").
     - **Code:** A unique identifier (e.g., "tshirt").
     - **Type:** Select **Selection** (for products with variants).
     - **Status:** Set to **Enabled**.

3. **Add General Information:**
   - Use the **Text** tab to add a description of the product.
   - Use the **Media** tab to upload images of the general product.

4. **Assign Attributes:**
   - Go to the **Attributes** tab.
   - Assign the relevant attributes (e.g., Size, Color).

5. **Save the Parent Product.**

---

### **3. Create Variant Products**
Each variant (e.g., "Red T-Shirt - Medium") is added as a child product.

#### **Steps:**
1. **Add a New Product:**
   - Click **+ Add** to create a new product for each variant.

2. **Fill in the Details:**
   - **Name:** The variant name (e.g., "Red T-Shirt - Medium").
   - **Code:** A unique identifier (e.g., "tshirt-red-medium").
   - **Type:** Select **Variant**.
   - **Status:** Set to **Enabled**.

3. **Assign Attributes for the Variant:**
   - Go to the **Attributes** tab.
   - Assign specific attribute values (e.g., Size: Medium, Color: Red).

4. **Set Pricing:**
   - Go to the **Price** tab and set the price for this variant.

5. **Add Images (Optional):**
   - Use the **Media** tab to upload images specific to the variant.

6. **Set Stock Levels:**
   - Go to the **Stock** tab and configure the available stock for this variant.

7. **Link to the Parent Product:**
   - Go to the **Product → Product Reference** tab.
   - Select the parent product (e.g., "T-Shirt") as the reference.

8. **Save the Variant Product.**

---

### **4. Link Variants to Parent Product**
1. Open the parent product in the admin panel.
2. Go to the **Product → Product Reference** tab.
3. Link all child products (variants) to the parent product.

---

### **5. Test Your Variants**
1. **Frontend Store:**
   - Visit your shop (e.g., `http://localhost:8000/shop`) to view the parent product.
   - Check the variant selection dropdowns for attributes like size and color.
2. **Ensure Variants Display Correctly:**
   - Ensure the stock, price, and attributes update dynamically when different variants are selected.

---

### **Tips:**
- Use clear and consistent naming conventions for variants to avoid confusion.
- Regularly monitor stock levels for each variant to ensure availability.
- Optimize images for each variant to enhance customer experience.

Would you like to explore advanced variant options or automation tools?