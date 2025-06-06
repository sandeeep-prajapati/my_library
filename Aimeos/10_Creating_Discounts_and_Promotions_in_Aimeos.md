### **Implementing Discount Rules, Voucher Codes, and Promotional Pricing in Aimeos**

Aimeos provides a flexible system for creating and managing discounts, voucher codes, and promotional pricing, which can be easily customized to suit your eCommerce store. Below is a guide to implementing these features in your store:

---

### **1. Discount Rules**

Aimeos allows you to set up various discount rules that can be applied automatically to orders, products, or customers. Discounts can be based on factors such as order total, product categories, quantity, or user groups.

#### **Steps to Implement Discount Rules:**

1. **Creating Discount Rules in the Admin Panel:**
   - Aimeos includes a built-in interface in the admin panel for creating and managing discount rules.
   - Go to the **Admin Panel > Discounts** section to create new discount rules.
   - You can define rules based on:
     - **Discount Type**: Fixed amount or percentage off
     - **Conditions**: Minimum order value, specific products, or categories
     - **Validity Period**: Define the start and end dates for the discount
     - **Usage Limit**: Limit the number of times the discount can be used per customer or globally

2. **Creating Discount Rule Example:**
   - Example of a simple discount rule: "Get 10% off on orders over $100."
     - **Condition**: Order total greater than $100
     - **Discount Type**: Percentage (10%)
   - This rule will apply to eligible orders automatically during checkout.

3. **Custom Discount Logic:**
   - If you need custom logic for applying discounts, you can extend the discount functionality by writing your own conditions or discount calculations.
   - Example: A discount rule based on a custom field like a membership level:
     ```php
     public function applyDiscount($order)
     {
         if ($order->customer->membership_level === 'premium') {
             $order->applyDiscount(0.2); // Apply 20% discount for premium members
         }
     }
     ```

---

### **2. Voucher Codes**

Voucher codes (also known as coupon codes) allow customers to apply a specific code to receive a discount during checkout. Aimeos provides a built-in voucher system that can be extended to meet your needs.

#### **Steps to Implement Voucher Codes:**

1. **Creating Voucher Codes in the Admin Panel:**
   - In the **Admin Panel > Discounts** section, you can create voucher codes by enabling the "Voucher" discount type.
   - Define the voucher code, its value (fixed or percentage), conditions (e.g., applicable products or categories), and usage limits.
   
2. **Voucher Code Setup Example:**
   - Example: "SAVE10" voucher code, which gives customers a 10% discount on orders over $50.
     - **Code**: SAVE10
     - **Discount**: 10% off
     - **Condition**: Order total greater than $50
     - **Usage Limit**: 1 use per customer

3. **Voucher Code Application:**
   - Customers will enter the voucher code during the checkout process in a designated "Voucher Code" field. The system will validate the code and apply the appropriate discount if it meets the conditions.

4. **Custom Voucher Validation:**
   - You can create custom validation logic for voucher codes by extending the `Voucher` model.
   - Example: Check if the voucher code is valid only for new customers:
     ```php
     public function validateVoucher($voucherCode)
     {
         $voucher = Voucher::findByCode($voucherCode);
         if ($voucher && $voucher->customer->isNewCustomer()) {
             return true;
         }
         return false;
     }
     ```

---

### **3. Promotional Pricing**

Promotional pricing allows you to offer temporary discounts on products or categories, which can be applied automatically or with a voucher code.

#### **Steps to Implement Promotional Pricing:**

1. **Create Promotional Price in Admin Panel:**
   - In the **Admin Panel > Products** section, you can create promotional prices for specific products.
   - You can define a temporary discount for a product that will automatically apply during the promotion period.

2. **Example: Setting a Promotional Price for a Product:**
   - If you want to offer a 20% discount on a product, you can set a new price for the product, or set the discount percentage.
     - **Product Price**: $100
     - **Promotional Price**: $80 (20% off)
   - You can configure this by setting a **Start Date** and **End Date** for the promotion.

3. **Custom Promotional Pricing Logic:**
   - You can also create complex promotional pricing logic, such as "Buy 1, Get 1 Free" or a discount based on customer groups.
   - Example: If a customer buys a specific product, they get a discount on another product:
     ```php
     public function applyPromotions($order)
     {
         $product = $order->getProductById(123); // Product ID
         if ($product && $order->getTotal() > 100) {
             $order->applyDiscount(0.5, $product->id); // 50% off on the next product
         }
     }
     ```

4. **Promotional Pricing with Customer Groups:**
   - You can set up promotions for specific customer groups, such as giving a higher discount to VIP customers. You can define this logic in the `applyPromotions()` method or in the admin panel.

---

### **4. Managing Discount and Voucher Code Rules Programmatically**

You may want to manage discounts, voucher codes, and promotions programmatically (e.g., adding them via a custom form or bulk import).

#### **Steps for Programmatic Discount Management:**

1. **Create Discounts Programmatically:**
   - You can create discount rules programmatically using Aimeos' API. For example, to create a percentage discount on orders over $100:
     ```php
     use Aimeos\MShop\Discount\Item\Manager;

     $manager = \Aimeos\MShop::create('discount');
     $rule = $manager->createItem([
         'type' => 'percent',
         'value' => 10,
         'minorder' => 100,
     ]);
     $manager->saveItem($rule);
     ```

2. **Applying Discounts to Cart Programmatically:**
   - You can apply discounts to the cart based on specific conditions:
     ```php
     use Aimeos\Shop\Facades\Shop;

     $cart = Shop::cart();
     $cart->addDiscount(10); // Apply a 10% discount
     ```

3. **Voucher Code Validation Programmatically:**
   - Validate and apply voucher codes in the controller when the customer submits a code during checkout:
     ```php
     use Aimeos\Shop\Facades\Shop;

     public function applyVoucher(Request $request)
     {
         $voucherCode = $request->input('voucher_code');
         $voucher = Shop::voucher()->findByCode($voucherCode);

         if ($voucher) {
             $cart = Shop::cart();
             $cart->addVoucher($voucher);
             return redirect()->route('checkout')->with('status', 'Voucher applied!');
         }

         return redirect()->back()->withErrors('Invalid voucher code');
     }
     ```

---

### **5. Testing and Debugging Discounts, Vouchers, and Promotions**

After implementing discounts, voucher codes, and promotional pricing, it’s essential to thoroughly test these features:

1. **Test Discount Rules:**
   - Verify that the discounts are applied correctly based on the conditions (e.g., total order value, specific products).
   
2. **Test Voucher Codes:**
   - Ensure that voucher codes are correctly validated and applied during the checkout process.

3. **Test Promotional Pricing:**
   - Make sure the promotional pricing is applied during the specified period and that customers can see the correct prices.

4. **Debugging:**
   - Use Aimeos' logging and debugging features to trace any issues with discounts, vouchers, or promotions:
     ```php
     Log::debug('Voucher applied:', ['voucher' => $voucherCode]);
     ```

---

### **Conclusion**

Aimeos offers powerful tools for creating and managing discounts, voucher codes, and promotional pricing. Whether you want to create simple percentage discounts, apply complex promotional logic, or manage custom voucher codes, you can do so easily using the built-in features or by extending them programmatically.